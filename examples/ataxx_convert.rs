use std::{
    env,
    fs::File,
    io::{BufWriter, Read, Write},
};

const SOURCE_SIZE: usize = 20;

fn to49(bb64: u64) -> u64 {
    let mut out = 0u64;
    for rank in 0..7 {
        let row = (bb64 >> (8 * rank)) & 0x7F;
        out |= row << (7 * rank);
    }
    out
}

struct Converted {
    stm: u64,
    ntm: u64,
    score: i16,
    result: u8,
    origstm: u8,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let input = args.get(1).cloned().unwrap_or("C:/shared/ataxx/data/Zataxx-550M.bin".to_string());
    let output = args.get(2).cloned().unwrap_or("C:/shared/ataxx/data/ataxx-bullet.data".to_string());
    let limit: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(usize::MAX);

    let meta = std::fs::metadata(&input).unwrap();
    let total = ((meta.len() as usize) / SOURCE_SIZE).min(limit);
    println!("Converting {total} entries from {input}");

    let mut file = File::open(&input).unwrap();
    let mut boards: Vec<Converted> = Vec::with_capacity(total);

    let chunk_entries = 1 << 21;
    let mut raw = vec![0u8; chunk_entries * SOURCE_SIZE];

    let mut score_sums = [[0f64; 3]; 2];
    let mut score_counts = [[0u64; 3]; 2];

    while boards.len() < total {
        let want = (total - boards.len()).min(chunk_entries) * SOURCE_SIZE;
        let mut got = 0;
        while got < want {
            let count = file.read(&mut raw[got..want]).unwrap();
            if count == 0 {
                break;
            }
            got += count;
        }
        if got == 0 {
            break;
        }

        for entry in 0..got / SOURCE_SIZE {
            let bytes = &raw[entry * SOURCE_SIZE..];
            let white = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
            let black = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
            let turn = bytes[16];
            let wdl = bytes[17];
            let score = i16::from_le_bytes(bytes[18..20].try_into().unwrap());

            let (stm, ntm, result) = if turn == 0 { (white, black, wdl) } else { (black, white, 2 - wdl) };

            score_sums[turn as usize][result as usize] += f64::from(score);
            score_counts[turn as usize][result as usize] += 1;

            boards.push(Converted { stm: to49(stm), ntm: to49(ntm), score, result, origstm: turn });
        }

        println!("Converted {}", boards.len());
    }

    let avg = |turn: usize, result: usize| score_sums[turn][result] / score_counts[turn][result].max(1) as f64;
    let white_corr = avg(0, 2) - avg(0, 0);
    let black_corr = avg(1, 2) - avg(1, 0);
    println!("Score-result correlation: white-to-move {white_corr:.1}, black-to-move {black_corr:.1}");

    if white_corr > 0.0 && black_corr > 0.0 {
        println!("Scores are stm-relative, no adjustment");
    } else if white_corr > 0.0 && black_corr < 0.0 {
        println!("Scores appear white-relative, negating black-to-move entries");
        for board in boards.iter_mut() {
            if board.origstm == 1 {
                board.score = -board.score;
            }
        }
    } else if white_corr < 0.0 && black_corr > 0.0 {
        println!("Scores appear black-relative, negating white-to-move entries");
        for board in boards.iter_mut() {
            if board.origstm == 0 {
                board.score = -board.score;
            }
        }
    } else {
        println!("Scores appear ntm-relative, negating all entries");
        for board in boards.iter_mut() {
            board.score = -board.score;
        }
    }

    println!("Shuffling {} entries", boards.len());
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next_random = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for i in (1..boards.len()).rev() {
        let j = (next_random() % (i as u64 + 1)) as usize;
        boards.swap(i, j);
    }

    println!("Writing {output}");
    let mut out = BufWriter::with_capacity(1 << 24, File::create(&output).unwrap());
    for board in &boards {
        out.write_all(&board.stm.to_le_bytes()).unwrap();
        out.write_all(&board.ntm.to_le_bytes()).unwrap();
        out.write_all(&0u64.to_le_bytes()).unwrap();
        out.write_all(&board.score.to_le_bytes()).unwrap();
        out.write_all(&[board.result, board.origstm]).unwrap();
        out.write_all(&0u16.to_le_bytes()).unwrap();
        out.write_all(&[0u8, 0u8]).unwrap();
    }
    out.flush().unwrap();
    println!("Done: {} entries, {} bytes", boards.len(), boards.len() * 32);
}
