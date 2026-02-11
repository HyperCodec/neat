use neat::*;

struct Genome {
    brain: NeuralNetwork<10, 4>,
    stats: PhysicalStats,
}

struct PhysicalStats {
    speed: f32,
    sight_range: u32,
}

struct Organism {
    genome: Genome,
    energy: f32,
}

fn main() {
    todo!("use NeuralNetwork along with other genomes for more complex organisms");
}
