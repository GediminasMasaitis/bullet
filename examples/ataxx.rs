/*
Trains the Ataxx.cpp NNUE and writes checkpoints in exactly the format that
`EvaluationNnueBase::init()` expects, so bullet can replace the LibTorch
trainer in `AtaxxDotCpp/trainer/main.cpp`.

Architecture (must match the engine, which has these sizes baked in):

    (49x2 -> 768)x2 -> 1, ReLU

  * `l0` is shared between the two perspectives. Within a perspective, block 0
    is "us", block 1 is "them", giving 98 inputs.
  * The two accumulators are concatenated, so `l1` is a 1536 -> 1 affine whose
    first 768 weights are the stm set (`hidden_weightses[0]`) and whose last
    768 are the ntm set (`hidden_weightses[1]`).
  * ReLU, *not* SCReLU - `evaluation_nn.cpp` clamps at zero only.

Output files, written to OUT_DIR every SAVE_RATE superbatches. Each run stamps
its start time into the name so consecutive runs never overwrite each other:

  * `bullet-<run>-sb<n>.nnue-floats` - raw f32, the file the engine actually reads
  * `bullet-<run>-sb<n>.nnue`        - i16 quantised by QUANT (side file, unused
                                       by the engine, kept for parity with the
                                       old trainer)
  * `bullet-<run>-sb<n>.txt`         - human readable dump of the quantised values

The engine quantises the floats itself by 128 at load time, and divides the
output by 32, so an eval of 1.0 in the loss below lands at 128 * 128 / 32 = 512
engine centipawns. That is the same 512 scale `forward_no_sig` used.

Ataxx's rules are symmetric under the dihedral group of the square (moves are
defined by Chebyshev distance, which D4 preserves), so a rotated or reflected
board has the same value and the same side to move. The LibTorch trainer
exploited that by averaging the loss over all 8 orientations. `SymmetryExpander`
below reproduces it exactly: every record is emitted 8 times, once per element
of D4, and since 8 divides the batch size those copies share a batch - and a
batch gradient is a mean, so this is identical to averaging 8 losses per sample.
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

/// Engine `EvaluationNnueBase::hidden_size`.
const HL: usize = 768;
/// Engine `EvaluationNnueBase::input_size`, i.e. 2 * 49.
const INPUTS: usize = 98;
/// A 7x7 board can never have more than 49 occupied squares.
const NNZ: usize = 49;

/// Weight given to the search score rather than the game result. The LibTorch
/// trainer fit pure WDL, so 0.0 reproduces it; raise it to blend in the score.
const SCORE_WEIGHT: f32 = 0.0;
/// Sigmoid scale for the search score. Datagen scores come out of the engine
/// already on its own 512 scale, so this converts them straight back to a
/// win probability.
const EVAL_SCALE: f32 = 512.0;

/// Quantisation factor for the side `.nnue` file, matching the old trainer.
const QUANT: i16 = 512;

/// How many of the 8 D4 orientations to train each position in. 8 reproduces
/// the LibTorch trainer; 1 disables augmentation, for an A/B test. Samples per
/// second barely change - a batch is still a batch - but 8 of every batch's
/// slots now go to one position, so a superbatch covers 8x fewer distinct
/// positions and `end_superbatch` has to grow to match.
const SYMMETRIES: usize = 8;
/// Source records expanded at a time. Kept a multiple of the batch size once
/// multiplied by SYMMETRIES, so a record's orientations never straddle a batch.
const EXPAND_CHUNK: usize = 65_536;

const DATA_PATH: &str = "C:/shared/ataxx/data/ataxx-bullet.data";
const OUT_DIR: &str = "C:/shared/ataxx/nets/current";
const SAVE_RATE: usize = 1;

/// The 8 elements of D4 as permutations of the 49 board squares, indexed
/// `D4[symmetry][rank * 7 + file]`.
const D4: [[u8; 49]; 8] = {
    let mut table = [[0u8; 49]; 8];
    let mut sym = 0;
    while sym < 8 {
        let mut rank = 0;
        while rank < 7 {
            let mut file = 0;
            while file < 7 {
                let (r, f) = match sym {
                    0 => (rank, file),                 // identity
                    1 => (file, 6 - rank),             // rotate 90
                    2 => (6 - rank, 6 - file),         // rotate 180
                    3 => (6 - file, rank),             // rotate 270
                    4 => (file, rank),                 // transpose
                    5 => (6 - file, 6 - rank),         // anti-transpose
                    6 => (rank, 6 - file),             // mirror files
                    _ => (6 - rank, file),             // mirror ranks
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
///
/// `bbs[0]` and `bbs[1]` are the side-to-move and not-side-to-move pieces,
/// packed 7 bits per rank so that bit `7 * rank + file` is one square. That is
/// the same indexing as the engine's `square_to_index`. `bbs[2]` is unused
/// (there are no walls in the dataset).
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

/// Emits every source record once per D4 orientation, tagging which one in the
/// spare `extra` byte for the mapper to read. Expanding here rather than in the
/// graph keeps the model a plain two-input net and costs no disk.
#[derive(Clone)]
struct SymmetryExpander<R> {
    inner: R,
    symmetries: usize,
}

impl<R: DataReader<AtaxxBoard>> DataReader<AtaxxBoard> for SymmetryExpander<R> {
    fn read_chunks<F: FnMut(&[AtaxxBoard]) -> bool>(&self, skip_count: usize, mut f: F) {
        let mut expanded = Vec::with_capacity(EXPAND_CHUNK * self.symmetries);

        // `skip_count` counts expanded records, the inner reader counts source ones.
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

/// The ntm view is the stm view rotated 180 degrees, so every symmetry must
/// commute with that rotation for the `48 - stm_sq` below to stay correct.
/// True because rot180 is central in D4, but cheap enough to prove at startup.
fn check_symmetries() {
    for (i, sym) in D4.iter().enumerate() {
        for sq in 0..49 {
            assert_eq!(48 - usize::from(sym[sq]), usize::from(sym[48 - sq]), "D4[{i}] does not commute with rot180");
        }

        let mut seen = [false; 49];
        for &sq in sym {
            assert!(!seen[usize::from(sq)], "D4[{i}] is not a permutation");
            seen[usize::from(sq)] = true;
        }

        assert!(D4.iter().filter(|other| *other == sym).count() == 1, "D4[{i}] is duplicated");
    }
}

fn main() {
    check_symmetries();

    let inputs = ModelInputs::default()
        .add_sparse("stm", (INPUTS, 1), NNZ)
        .add_sparse("ntm", (INPUTS, 1), NNZ)
        .add_dense("target", (1, 1));

    let mapper = ModelInputsMapper::build(&inputs, |pos: &AtaxxBoard, _, ((stm, ntm), target)| {
        // The engine builds the black accumulator from `flip_square`, a 180
        // degree rotation, which is `48 - index` in 49-square index space. The
        // converter stores boards in absolute coordinates, so rotate here when
        // black is to move to line the stm perspective up with the engine's.
        // The ntm perspective is always the opposite rotation.
        let flip_stm = pos.origstm == 1;
        // Which D4 orientation this copy of the record represents.
        let sym = &D4[usize::from(pos.extra)];

        let mut i = 0;
        for (block, mut occ) in [(0i32, pos.bbs[0]), (1i32, pos.bbs[1])] {
            while occ > 0 {
                let sq = occ.trailing_zeros() as i32;
                occ &= occ - 1;

                let oriented = if flip_stm { 48 - sq } else { sq };
                let stm_sq = i32::from(sym[oriented as usize]);
                stm[i] = 49 * block + stm_sq;
                ntm[i] = 49 * (1 - block) + (48 - stm_sq);
                i += 1;
            }
        }

        // Every unused slot needs the sentinel: the sparse matmul skips
        // negative indices rather than stopping at the first one, and these
        // buffers are pooled and so arrive holding the previous batch's data.
        for slot in i..NNZ {
            stm[slot] = -1;
            ntm[slot] = -1;
        }

        // `result` is already stm-relative, 0/1/2 for loss/draw/win.
        let result = f32::from(pos.result) / 2.0;
        let score = 1.0 / (1.0 + (-f32::from(pos.score) / EVAL_SCALE).exp());
        target[0] = SCORE_WEIGHT * score + (1.0 - SCORE_WEIGHT) * result;
    });

    let defn = ModelDefinition::build(&inputs, |builder, ((stm, ntm), target)| {
        let l0 = builder.new_affine("l0", INPUTS, HL);
        let l1 = builder.new_affine("l1", 2 * HL, 1);

        let stm_hidden = l0.forward(stm).relu();
        let ntm_hidden = l0.forward(ntm).relu();
        let hidden_layer = stm_hidden.concat(ntm_hidden);
        let output = l1.forward(hidden_layer);

        let loss = output.sigmoid().squared_error(target).reduce_sum_batch();

        (Some(loss), vec![("output".to_string(), output)])
    });

    let weights = ModelWeights::new(&defn, 198273612);
    let device = DefaultDevice::new(0).unwrap();

    // Clipping to +/-1.98 keeps the engine's int16 accumulator well clear of
    // overflow once it quantises by 128 at load time.
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

    // An epoch is one pass over the *distinct* positions in the file. A
    // superbatch is a fixed number of samples, and with SYMMETRIES = 8 only an
    // eighth of those are distinct, so it takes 8x as many superbatches to get
    // through the data. Derived from the file rather than hardcoded so the two
    // can't drift apart.
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
    println!("Saving checkpoints to {OUT_DIR}/bullet-{run_id}-sb<n>.*");

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

/// UTC start time as `YYYYMMDD-HHMMSS`, taken once so every checkpoint from a
/// single run shares one prefix.
fn timestamp() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (hour, minute, second) = (rem / 3600, rem % 3600 / 60, rem % 60);

    // Civil date from days since the epoch, per Howard Hinnant's `civil_from_days`.
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

/// The order the engine reads in `EvaluationNnueBase::init()`:
///
///   1. `input_weights`, looped hidden-major then input-major, i.e. row-major
///      over the 768x98 matrix. Bullet stores weights column-major, hence the
///      transpose. This is the same layout as LibTorch's flattened `fc1.weight`.
///   2. `input_biases`, 768 values (`fc1.bias`).
///   3. `hidden_weightses[0]` then `[1]`, 768 values each. `concat` puts stm
///      first, which is the order the engine reads them in (`fc2.weight`).
///   4. `hidden_bias`, one value (`fc2.bias`).
fn save_format() -> [SavedFormat; 4] {
    [SavedFormat::id("l0w").transpose(), SavedFormat::id("l0b"), SavedFormat::id("l1w"), SavedFormat::id("l1b")]
}

fn quantised_format() -> [SavedFormat; 4] {
    save_format().map(|fmt| fmt.round().quantise::<i16>(QUANT))
}

fn save(weights: &ModelWeights, run_id: &str, superbatch: usize) {
    let stem = format!("{OUT_DIR}/bullet-{run_id}-sb{superbatch}");

    let float_path = format!("{stem}.nnue-floats");
    let floats = weights.to_quantised_buffer(&save_format(), false).unwrap();
    fs::write(&float_path, &floats).unwrap();
    println!("Wrote {} ({} bytes)", float_path, floats.len());

    // The engine ignores these two, but the old trainer wrote them, so keep
    // them around. Quantisation can legitimately fail, which is not a reason
    // to interrupt training.
    let quantised = match weights.to_quantised_buffer(&quantised_format(), false) {
        Ok(buf) => buf,
        Err(e) => {
            println!("Skipping quantised output for superbatch {superbatch}: {e}");
            return;
        }
    };

    fs::write(format!("{stem}.nnue"), &quantised).unwrap();

    let names = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"];
    let mut human = BufWriter::new(File::create(format!("{stem}.txt")).unwrap());
    let mut offset = 0;
    for (name, fmt) in names.iter().zip(quantised_format()) {
        let section = fmt.write_to_byte_buffer(weights).unwrap();
        writeln!(human, "{name}").unwrap();
        for value in section.chunks_exact(2) {
            write!(human, "{} ", i16::from_le_bytes([value[0], value[1]])).unwrap();
        }
        writeln!(human).unwrap();
        offset += section.len();
    }
    assert_eq!(offset, quantised.len());
    human.flush().unwrap();
}
