/*
Trains the move-ordering policy network for Ataxx.cpp, in the layout
`PolicyNn::init()` reads. Build the engine with `DO_POLICY 1`.

    2916 -> 128 -> 529, SCReLU, softmax cross-entropy

Inputs are the same 36 overlapping 2x2 tuples the value network uses. Outputs
are one logit per move:

    0..48    drops, indexed by destination square
    49..528  jumps, enumerated per from-square over its on-board far targets

Everything is in absolute board coordinates with no perspective flip - the
input is already side-to-move relative (us / them), so a position and its
colour-mirror never collide. That is the bug that sank the old `policy` branch:
it flipped the input for black but not the move label, so the same input
carried two contradictory targets.

Data is `Zataxx-550M-bestmove-trimmed.bin`, 22 bytes per record: the 20-byte value
record plus the played `from` and `to`. Squares are 8 to a row (rank * 8 +
file), with 63 meaning "drop" in `from`.

Output goes to OUT_DIR as `policy-<run>-sb<n>.nnue-floats`; copy the one you
want to `AtaxxDotCpp/src/networks/policy.nnue-floats`. Linux builds embed it at
compile time, so rebuild after copying.
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
const INPUTS: usize = TUPLE_COUNT * PER_TUPLE;
/// Engine `PolicyNn::hidden_size`.
const HL: usize = 128;
/// Engine `PolicyNn::output_size`: 49 drops + 480 on-board jumps.
const OUTPUTS: usize = 529;
const NNZ: usize = TUPLE_COUNT;

const QA: i16 = 255;
const QB: i16 = 64;

const SYMMETRIES: usize = 8;
const EXPAND_CHUNK: usize = 65_536;

const DATA_PATH: &str = "C:/shared/ataxx/data/Zataxx-550M-bestmove-trimmed.bin";
const OUT_DIR: &str = "C:/shared/ataxx/nets/current";
const SAVE_RATE: usize = 1;

const NO_SQUARE: u8 = 63;

/// D4 as permutations of the 64-square board, `D4[symmetry][rank * 8 + file]`.
/// Off-board squares map to themselves, so `NO_SQUARE` survives a transform and
/// keeps meaning "drop".
const D4: [[u8; 64]; 8] = {
    let mut table = [[0u8; 64]; 8];
    let mut sym = 0;
    while sym < 8 {
        let mut square = 0;
        while square < 64 {
            table[sym][square] = square as u8;
            square += 1;
        }

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
                table[sym][rank * 8 + file] = (r * 8 + f) as u8;
                file += 1;
            }
            rank += 1;
        }
        sym += 1;
    }
    table
};

/// Move -> output index, mirroring `PolicyIndexClass` in the engine.
/// `POLICY_INDEX[from][to]`, with -1 where no move exists.
const POLICY_INDEX: [[i16; 64]; 64] = {
    let mut table = [[-1i16; 64]; 64];
    let mut index = 0i16;

    let mut rank = 0;
    while rank < 7 {
        let mut file = 0;
        while file < 7 {
            table[NO_SQUARE as usize][rank * 8 + file] = index;
            index += 1;
            file += 1;
        }
        rank += 1;
    }

    // Jumps: the ring at Chebyshev distance exactly 2, clipped to the board,
    // enumerated in ascending square order to match the engine's bitboard scan.
    let mut from_rank = 0i32;
    while from_rank < 7 {
        let mut from_file = 0i32;
        while from_file < 7 {
            let mut to_rank = 0i32;
            while to_rank < 7 {
                let mut to_file = 0i32;
                while to_file < 7 {
                    let dr = if to_rank > from_rank { to_rank - from_rank } else { from_rank - to_rank };
                    let df = if to_file > from_file { to_file - from_file } else { from_file - to_file };
                    let dist = if dr > df { dr } else { df };
                    if dist == 2 {
                        table[(from_rank * 8 + from_file) as usize][(to_rank * 8 + to_file) as usize] = index;
                        index += 1;
                    }
                    to_file += 1;
                }
                to_rank += 1;
            }
            from_file += 1;
        }
        from_rank += 1;
    }

    assert!(index as usize == OUTPUTS);
    table
};

/// A 22-byte record. Held as raw bytes because `#[repr(C)]` on the natural
/// field layout would pad it to 24 and break the fixed-size reader.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyEntry {
    pub bytes: [u8; 22],
}

unsafe impl FixedSizeData for PolicyEntry {}

impl PolicyEntry {
    fn white(&self) -> u64 {
        u64::from_le_bytes(self.bytes[0..8].try_into().unwrap())
    }
    fn black(&self) -> u64 {
        u64::from_le_bytes(self.bytes[8..16].try_into().unwrap())
    }
    fn turn(&self) -> u8 {
        self.bytes[16]
    }
    fn from(&self) -> u8 {
        self.bytes[20]
    }
    fn to(&self) -> u8 {
        self.bytes[21]
    }
    /// The spare `wdl` byte carries the D4 orientation, as `extra` does for the
    /// value trainers. The policy loss does not use the game result.
    fn symmetry(&self) -> usize {
        usize::from(self.bytes[17])
    }
    fn set_symmetry(&mut self, sym: usize) {
        self.bytes[17] = sym as u8;
    }
}

#[derive(Clone)]
struct SymmetryExpander<R> {
    inner: R,
    symmetries: usize,
}

impl<R: DataReader<PolicyEntry>> DataReader<PolicyEntry> for SymmetryExpander<R> {
    fn read_chunks<F: FnMut(&[PolicyEntry]) -> bool>(&self, skip_count: usize, mut f: F) {
        let mut expanded = Vec::with_capacity(EXPAND_CHUNK * self.symmetries);

        self.inner.read_chunks(skip_count / self.symmetries, |chunk| {
            for group in chunk.chunks(EXPAND_CHUNK) {
                expanded.clear();
                for pos in group {
                    for sym in 0..self.symmetries {
                        let mut copy = *pos;
                        copy.set_symmetry(sym);
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

fn permute(mut occ: u64, sym: &[u8; 64]) -> u64 {
    let mut out = 0u64;
    while occ > 0 {
        let square = occ.trailing_zeros() as usize;
        occ &= occ - 1;
        out |= 1u64 << sym[square];
    }
    out
}

fn main() {
    check_tables();

    let inputs = ModelInputs::default().add_sparse("stm", (INPUTS, 1), NNZ).add_dense("target", (OUTPUTS, 1));

    let mapper = ModelInputsMapper::build(&inputs, |pos: &PolicyEntry, _, (stm, target)| {
        // Unlike the value net, the *label* has to move with the board.
        let sym = &D4[pos.symmetry()];
        let (white, black) = (permute(pos.white(), sym), permute(pos.black(), sym));
        let (us, them) = if pos.turn() == 0 { (white, black) } else { (black, white) };

        const POWERS: [i32; 4] = [1, 3, 9, 27];
        // 8 squares to a row, so a 2x2 window covers offsets 0, 1, 8 and 9.
        const WINDOW: u64 = (1 << 0) | (1 << 1) | (1 << 8) | (1 << 9);

        let mut i = 0;
        for rank in 0..TUPLE_SIDE {
            for file in 0..TUPLE_SIDE {
                let offset = rank * 8 + file;
                let mut index = (PER_TUPLE * (rank * TUPLE_SIDE + file)) as i32;

                let mut ours = (us >> offset) & WINDOW;
                while ours > 0 {
                    let bit = ours.trailing_zeros() as usize;
                    ours &= ours - 1;
                    index += POWERS[if bit > 1 { bit - 6 } else { bit }];
                }

                let mut theirs = (them >> offset) & WINDOW;
                while theirs > 0 {
                    let bit = theirs.trailing_zeros() as usize;
                    theirs &= theirs - 1;
                    index += 2 * POWERS[if bit > 1 { bit - 6 } else { bit }];
                }

                stm[i] = index;
                i += 1;
            }
        }

        let from = sym[usize::from(pos.from())];
        let to = sym[usize::from(pos.to())];
        let label = POLICY_INDEX[usize::from(from)][usize::from(to)];

        // The buffer is pooled, so clear it before writing the one-hot.
        target.fill(0.0);
        if label >= 0 {
            target[label as usize] = 1.0;
        }
    });

    let defn = ModelDefinition::build(&inputs, |builder, (stm, target)| {
        let l0 = builder.new_affine("l0", INPUTS, HL);
        let l1 = builder.new_affine("l1", HL, OUTPUTS);

        let logits = l1.forward(l0.forward(stm).screlu());
        // The loss is per-output, so collapse the 529 rows to a scalar per
        // sample before summing over the batch.
        let loss = logits.softmax_crossentropy_loss(target).reduce_sum_rows().reduce_sum_batch();

        (Some(loss), vec![("logits".to_string(), logits)])
    });

    let weights = ModelWeights::new(&defn, 198273612);
    let device = DefaultDevice::new(0).unwrap();
    let params = AdamWParams { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98 };
    let mut optimiser = AdamW::new(defn, weights, device, params).unwrap();

    // The one-hot target is 529 floats per sample, so batches move a lot more
    // host-to-device data than the value trainers do; hence the smaller batch.
    let batch_size = 8192;
    let batches_per_superbatch = 3052;
    let end_superbatch = 20;
    assert_eq!((EXPAND_CHUNK * SYMMETRIES) % batch_size, 0, "a record's orientations must share a batch");

    let reader = SymmetryExpander { inner: FixedSizeDataReader::new(&[DATA_PATH]), symmetries: SYMMETRIES };
    let loader = ReadMapLoader::new(reader, mapper, 4);

    let schedule = TrainingSchedule {
        steps: TrainingSteps { batch_size, batches_per_superbatch, start_superbatch: 1, end_superbatch },
        log_rate: 128,
        lr_schedule: Box::new(|step| if step.superbatch() > 15 { 0.0001 } else { 0.001 }),
    };

    let positions = fs::metadata(DATA_PATH).unwrap().len() as usize / size_of::<PolicyEntry>();
    let distinct_per_superbatch = batch_size * batches_per_superbatch / SYMMETRIES;
    let superbatches_per_epoch = positions as f32 / distinct_per_superbatch as f32;
    let total_epochs = end_superbatch as f32 / superbatches_per_epoch;

    println!(
        "Dataset: {} labelled positions x {SYMMETRIES} orientation(s); 1 epoch = {} superbatches",
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
    println!("Saving checkpoints to {OUT_DIR}/policy-{run_id}-sb<n>.*");

    train(
        &mut optimiser,
        schedule,
        loader,
        |_, _, _| {},
        |optimiser, step| {
            let sb = step.superbatch();

            println!(
                "Epoch {} / {}",
                logger::ansi(format!("{:.2}", sb as f32 / superbatches_per_epoch), logger::num_cs()),
                logger::ansi(format!("{total_epochs:.2}"), logger::num_cs()),
            );

            if sb % SAVE_RATE == 0 {
                let weights = optimiser.cpu_weights().unwrap();
                save(&weights, &run_id, sb);
            }
        },
    )
    .unwrap();
}

/// The index table and the symmetries have to agree with the engine exactly,
/// and both are generated rather than written out - so prove the invariants.
fn check_tables() {
    let mut seen = [false; OUTPUTS];
    let mut count = 0;
    for from in 0..64usize {
        for to in 0..64usize {
            let idx = POLICY_INDEX[from][to];
            if idx >= 0 {
                assert!(!seen[idx as usize], "index {idx} assigned twice");
                seen[idx as usize] = true;
                count += 1;
            }
        }
    }
    assert_eq!(count, OUTPUTS, "index table must be a bijection onto 0..{OUTPUTS}");
    assert_eq!(POLICY_INDEX[NO_SQUARE as usize][NO_SQUARE as usize], -1, "passes must have no index");

    // A symmetry must map legal moves to legal moves, or the transformed label
    // would land on -1 and silently train an all-zero target.
    for (s, sym) in D4.iter().enumerate() {
        assert_eq!(sym[NO_SQUARE as usize], NO_SQUARE, "D4[{s}] must fix the drop marker");
        for from in 0..64usize {
            for to in 0..64usize {
                if POLICY_INDEX[from][to] >= 0 {
                    let (f, t) = (usize::from(sym[from]), usize::from(sym[to]));
                    assert!(POLICY_INDEX[f][t] >= 0, "D4[{s}] maps a legal move to an illegal one");
                }
            }
        }
    }
}

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

/// The order `PolicyNn::init()` reads in. `l0w` stays column-major (which is
/// input-major, what the accumulator wants); `l1w` is transposed to move-major
/// so scoring one move reads a contiguous run of 128.
fn save_format() -> [SavedFormat; 4] {
    [
        SavedFormat::id("l0w"),
        SavedFormat::id("l0b"),
        SavedFormat::id("l1w").transpose(),
        SavedFormat::id("l1b"),
    ]
}

fn quantised_format() -> [SavedFormat; 4] {
    [
        SavedFormat::id("l0w").round().quantise::<i16>(QA),
        SavedFormat::id("l0b").round().quantise::<i16>(QA),
        SavedFormat::id("l1w").transpose().round().quantise::<i16>(QB),
        SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
    ]
}

fn check_engine_headroom(weights: &ModelWeights, superbatch: usize) {
    let format = [SavedFormat::id("l1w").round().quantise::<i16>(QB)];
    let Ok(buf) = weights.to_quantised_buffer(&format, false) else {
        println!("WARNING sb{superbatch}: output weights exceed i16 at QB");
        return;
    };

    // Only one move's row is summed at a time, so the bound is per row.
    let rows: Vec<i64> = buf
        .chunks_exact(2)
        .map(|b| i64::from(i16::from_le_bytes([b[0], b[1]]).abs()))
        .collect::<Vec<_>>()
        .chunks(HL)
        .map(|r| r.iter().sum())
        .collect();
    let worst_case = rows.iter().copied().max().unwrap_or(0) * i64::from(QA) * i64::from(QA);
    let headroom = f64::from(i32::MAX) / worst_case.max(1) as f64;

    if headroom < 1.0 {
        println!("WARNING sb{superbatch}: policy score can overflow int32 ({worst_case}, {headroom:.2}x)");
    } else if superbatch == 1 || headroom < 2.0 {
        println!("Engine int32 headroom: {headroom:.1}x worst case");
    }
}

fn save(weights: &ModelWeights, run_id: &str, superbatch: usize) {
    let stem = format!("{OUT_DIR}/policy-{run_id}-sb{superbatch}");

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
