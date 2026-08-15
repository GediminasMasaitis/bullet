/*
Trains the Zataxx architecture for Ataxx.cpp and writes it in the layout
`EvaluationNnueTuple::init()` reads. Build the engine with `TUPLE_NNUE 1`.

    2916 -> 256 -> 1, SCReLU

  * The 2916 inputs are 36 overlapping 2x2 windows of the board, one per
    top-left square of a 6x6 grid. Each window's four squares are empty / ours
    / theirs, giving 3^4 = 81 states per window, and a window always
    contributes exactly one feature - so the input is always exactly 36-hot,
    including the all-empty state.
  * *Single perspective*: features are already relative to the side to move,
    so there is one HL-wide accumulator, not a stm/ntm pair, and no 180
    degree board flip anywhere. `l1` is therefore HL -> 1, not 2*HL -> 1.
  * The two layers quantise by different factors (QA for the input layer, QB
    for the output), so unlike the 98 -> 768 network they cannot share one.

Compared with Zataxx itself the only deliberate difference is the eval scale:
Zataxx uses 400, the engine here uses 512 to stay on the scale its search
margins are tuned against. Change `scale` in evaluation_nn_tuple.h to match
Zataxx exactly.

Output goes to OUT_DIR as `tuple-<run>-sb<n>.nnue-floats`; copy the one you
want to `AtaxxDotCpp/src/networks/default-tuple.nnue-floats`. Linux builds
embed it at compile time, so rebuild after copying.
*/
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    time::{SystemTime, UNIX_EPOCH},
};

use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat},
    optimiser::adam::{AdamW, AdamWParams},
    reader::{DataReader, FixedSizeData, FixedSizeDataReader, ReadMapLoader},
    run::{DefaultDevice, TrainingSchedule, TrainingSteps, logger, train},
};

const TUPLE_SIDE: usize = 6;
const TUPLE_COUNT: usize = TUPLE_SIDE * TUPLE_SIDE;
const PER_TUPLE: usize = 81;
/// Engine `EvaluationNnueTuple::input_size`.
const INPUTS: usize = TUPLE_COUNT * PER_TUPLE;
/// Engine `hidden_size` - must match `EvaluationNnueTuple::hidden_size`.
const HL: usize = 256;
/// Every window contributes exactly one feature, so this is exact, not a bound.
const NNZ: usize = TUPLE_COUNT;

/// Engine quantisation factors. The input layer uses QA, the output layer QB,
/// and the output bias their product.
const QA: i16 = 255;
const QB: i16 = 64;

const SCORE_WEIGHT: f32 = 0.0;
const EVAL_SCALE: f32 = 512.0;

const SYMMETRIES: usize = 8;
const EXPAND_CHUNK: usize = 65_536;

const DATA_PATH: &str = "C:/shared/ataxx/data/ataxx-bullet.data";
const OUT_DIR: &str = "C:/shared/ataxx/nets/current";
const SAVE_RATE: usize = 1;

/// The 8 elements of D4 as permutations of the 49 board squares.
const D4: [[u8; 49]; 8] = {
    let mut table = [[0u8; 49]; 8];
    let mut sym = 0;
    while sym < 8 {
        let mut rank = 0;
        while rank < 7 {
            let mut file = 0;
            while file < 7 {
                let (r, f) = match sym {
                    0 => (rank, file),
                    1 => (file, 6 - rank),
                    2 => (6 - rank, 6 - file),
                    3 => (6 - file, rank),
                    4 => (file, rank),
                    5 => (6 - file, 6 - rank),
                    6 => (rank, 6 - file),
                    _ => (6 - rank, file),
                };
                table[sym][rank * 7 + file] = (r * 7 + f) as u8;
                file += 1;
            }
            rank += 1;
        }
        sym += 1;
    }
    table
};

/// One entry of `ataxx-bullet.data`, as produced by `ataxx_convert.rs`.
/// `bbs[0]` is the side to move, `bbs[1]` the opponent, 7 bits per rank.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AtaxxBoard {
    pub bbs: [u64; 3],
    pub score: i16,
    pub result: u8,
    pub origstm: u8,
    pub fullm: u16,
    pub halfm: u8,
    pub extra: u8,
}

unsafe impl FixedSizeData for AtaxxBoard {}

/// Emits every source record once per D4 orientation, tagged in `extra`.
#[derive(Clone)]
struct SymmetryExpander<R> {
    inner: R,
    symmetries: usize,
}

impl<R: DataReader<AtaxxBoard>> DataReader<AtaxxBoard> for SymmetryExpander<R> {
    fn read_chunks<F: FnMut(&[AtaxxBoard]) -> bool>(&self, skip_count: usize, mut f: F) {
        let mut expanded = Vec::with_capacity(EXPAND_CHUNK * self.symmetries);

        self.inner.read_chunks(skip_count / self.symmetries, |chunk| {
            for group in chunk.chunks(EXPAND_CHUNK) {
                expanded.clear();
                for pos in group {
                    for sym in 0..self.symmetries {
                        let mut copy = *pos;
                        copy.extra = sym as u8;
                        expanded.push(copy);
                    }
                }

                if f(&expanded) {
                    return true;
                }
            }

            false
        });
    }
}

/// Applies a square permutation to a 49-bit board.
fn permute(mut occ: u64, sym: &[u8; 49]) -> u64 {
    let mut out = 0u64;
    while occ > 0 {
        let square = occ.trailing_zeros() as usize;
        occ &= occ - 1;
        out |= 1u64 << sym[square];
    }
    out
}

fn main() {
    let inputs = ModelInputs::default().add_sparse("stm", (INPUTS, 1), NNZ).add_dense("target", (1, 1));

    let mapper = ModelInputsMapper::build(&inputs, |pos: &AtaxxBoard, _, (stm, target)| {
        // Symmetries act on the board, then features are derived - much
        // easier to get right than permuting 2916 feature indices, and the
        // tuple grid maps onto itself under D4 so the two agree.
        let sym = &D4[usize::from(pos.extra)];
        let us = permute(pos.bbs[0], sym);
        let them = permute(pos.bbs[1], sym);

        // No perspective flip: `us` / `them` are already side-to-move
        // relative, and this network has only the one perspective.
        const POWERS: [i32; 4] = [1, 3, 9, 27];
        // 7 bits per rank, so a 2x2 window covers offsets 0, 1, 7 and 8.
        const WINDOW: u64 = (1 << 0) | (1 << 1) | (1 << 7) | (1 << 8);

        let mut i = 0;
        for rank in 0..TUPLE_SIDE {
            for file in 0..TUPLE_SIDE {
                let offset = rank * 7 + file;
                let mut index = (PER_TUPLE * (rank * TUPLE_SIDE + file)) as i32;

                let mut ours = (us >> offset) & WINDOW;
                while ours > 0 {
                    let bit = ours.trailing_zeros() as usize;
                    ours &= ours - 1;
                    index += POWERS[if bit > 1 { bit - 5 } else { bit }];
                }

                let mut theirs = (them >> offset) & WINDOW;
                while theirs > 0 {
                    let bit = theirs.trailing_zeros() as usize;
                    theirs &= theirs - 1;
                    index += 2 * POWERS[if bit > 1 { bit - 5 } else { bit }];
                }

                stm[i] = index;
                i += 1;
            }
        }

        debug_assert_eq!(i, NNZ);

        let result = f32::from(pos.result) / 2.0;
        let score = 1.0 / (1.0 + (-f32::from(pos.score) / EVAL_SCALE).exp());
        target[0] = SCORE_WEIGHT * score + (1.0 - SCORE_WEIGHT) * result;
    });

    let defn = ModelDefinition::build(&inputs, |builder, (stm, target)| {
        let l0 = builder.new_affine("l0", INPUTS, HL);
        let l1 = builder.new_affine("l1", HL, 1);

        let output = l1.forward(l0.forward(stm).screlu());
        let loss = output.sigmoid().squared_error(target).reduce_sum_batch();

        (Some(loss), vec![("output".to_string(), output)])
    });

    let weights = ModelWeights::new(&defn, 198273612);
    let device = DefaultDevice::new(0).unwrap();

    let params = AdamWParams { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98 };
    let mut optimiser = AdamW::new(defn, weights, device, params).unwrap();

    let batch_size = 16_384;
    let batches_per_superbatch = 6104;
    let end_superbatch = 44;
    assert_eq!((EXPAND_CHUNK * SYMMETRIES) % batch_size, 0, "a record's orientations must share a batch");

    let reader = SymmetryExpander { inner: FixedSizeDataReader::new(&[DATA_PATH]), symmetries: SYMMETRIES };
    let loader = ReadMapLoader::new(reader, mapper, 4);

    let schedule = TrainingSchedule {
        steps: TrainingSteps { batch_size, batches_per_superbatch, start_superbatch: 1, end_superbatch },
        log_rate: 128,
        lr_schedule: Box::new(|step| if step.superbatch() > 34 { 0.0001 } else { 0.001 }),
    };

    let positions = fs::metadata(DATA_PATH).unwrap().len() as usize / size_of::<AtaxxBoard>();
    let distinct_per_superbatch = batch_size * batches_per_superbatch / SYMMETRIES;
    let superbatches_per_epoch = positions as f32 / distinct_per_superbatch as f32;
    let total_epochs = end_superbatch as f32 / superbatches_per_epoch;

    println!(
        "Dataset: {} positions x {SYMMETRIES} orientation(s); 1 epoch = {} superbatches",
        logger::ansi(positions, logger::num_cs()),
        logger::ansi(format!("{superbatches_per_epoch:.1}"), logger::num_cs()),
    );
    println!(
        "Scheduled {} superbatches = {} epochs",
        logger::ansi(end_superbatch, logger::num_cs()),
        logger::ansi(format!("{total_epochs:.2}"), logger::num_cs()),
    );

    fs::create_dir_all(OUT_DIR).unwrap();
    let run_id = timestamp();
    println!("Saving checkpoints to {OUT_DIR}/tuple-{run_id}-sb<n>.*");

    train(
        &mut optimiser,
        schedule,
        loader,
        |_, _, _| {},
        |optimiser, step| {
            let sb = step.superbatch();

            println!(
                "Epoch {} / {} ({} distinct positions seen)",
                logger::ansi(format!("{:.2}", sb as f32 / superbatches_per_epoch), logger::num_cs()),
                logger::ansi(format!("{total_epochs:.2}"), logger::num_cs()),
                logger::ansi(sb * distinct_per_superbatch, logger::num_cs()),
            );

            if sb % SAVE_RATE == 0 {
                let weights = optimiser.cpu_weights().unwrap();
                save(&weights, &run_id, sb);
            }
        },
    )
    .unwrap();
}

/// UTC start time as `YYYYMMDD-HHMMSS`, taken once per run.
fn timestamp() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (hour, minute, second) = (rem / 3600, rem % 3600 / 60, rem % 60);

    let z = days as i64 + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = era * 400 + yoe + i64::from(month <= 2);

    format!("{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}")
}

/// The order `EvaluationNnueTuple::init()` reads in. `l0w` is *not* transposed:
/// the engine wants it input-major, which is already bullet's column-major
/// layout for a 256 x 2916 matrix.
fn save_format() -> [SavedFormat; 4] {
    [SavedFormat::id("l0w"), SavedFormat::id("l0b"), SavedFormat::id("l1w"), SavedFormat::id("l1b")]
}

/// The same weights quantised the way the engine will quantise them - each
/// layer by its own factor, and the output bias by both.
fn quantised_format() -> [SavedFormat; 4] {
    [
        SavedFormat::id("l0w").round().quantise::<i16>(QA),
        SavedFormat::id("l0b").round().quantise::<i16>(QA),
        SavedFormat::id("l1w").round().quantise::<i16>(QB),
        SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
    ]
}

/// The engine sums `clamp(acc, 0, QA)^2 * weight` into an int32. Check the
/// weights actually leave room for that rather than assuming they do.
fn check_engine_headroom(weights: &ModelWeights, superbatch: usize) {
    let format = [SavedFormat::id("l1w").round().quantise::<i16>(QB)];
    let Ok(buf) = weights.to_quantised_buffer(&format, false) else {
        println!("WARNING sb{superbatch}: output weights exceed i16 at QB");
        return;
    };

    let total: i64 = buf.chunks_exact(2).map(|b| i64::from(i16::from_le_bytes([b[0], b[1]]).abs())).sum();
    let worst_case = total * i64::from(QA) * i64::from(QA);
    let headroom = f64::from(i32::MAX) / worst_case.max(1) as f64;

    if headroom < 1.0 {
        println!("WARNING sb{superbatch}: engine eval can overflow int32 (worst case {worst_case}, {headroom:.2}x)");
    } else if superbatch == 1 || headroom < 2.0 {
        println!("Engine int32 headroom: {headroom:.1}x worst case");
    }
}

fn save(weights: &ModelWeights, run_id: &str, superbatch: usize) {
    let stem = format!("{OUT_DIR}/tuple-{run_id}-sb{superbatch}");

    check_engine_headroom(weights, superbatch);

    let float_path = format!("{stem}.nnue-floats");
    let floats = weights.to_quantised_buffer(&save_format(), false).unwrap();
    fs::write(&float_path, &floats).unwrap();
    println!("Wrote {} ({} bytes)", float_path, floats.len());

    let quantised = match weights.to_quantised_buffer(&quantised_format(), false) {
        Ok(buf) => buf,
        Err(e) => {
            println!("Skipping quantised output for superbatch {superbatch}: {e}");
            return;
        }
    };

    fs::write(format!("{stem}.nnue"), &quantised).unwrap();

    let names = ["l0.weight", "l0.bias", "l1.weight", "l1.bias"];
    let mut human = BufWriter::new(File::create(format!("{stem}.txt")).unwrap());
    for (name, fmt) in names.iter().zip(quantised_format()) {
        let section = fmt.write_to_byte_buffer(weights).unwrap();
        writeln!(human, "{name}").unwrap();
        for value in section.chunks_exact(2) {
            write!(human, "{} ", i16::from_le_bytes([value[0], value[1]])).unwrap();
        }
        writeln!(human).unwrap();
    }
    human.flush().unwrap();
}
