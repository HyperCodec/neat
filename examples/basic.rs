use neat::*;

// approximate the to_degrees function, which should be pretty
// hard for a traditional network to learn since it's not really close to -1..1 mapping.
fn fitness(net: &NeuralNetwork<1, 1>) -> f32 {
    let mut rng = rand::rng();
    let mut total_fitness = 0.0;

    // it's good practice to test on multiple inputs to get a more accurate fitness score
    for _ in 0..100 {
        let input = rng.random_range(-10.0..10.0);
        let output = net.predict([input])[0];
        let expected_output = input.to_degrees();

        // basically just using negative error as fitness.
        // percentage error doesn't work as well here since
        // expected_output can be either very small or very large in magnitude.
        total_fitness -= (output - expected_output).abs();
    }

    total_fitness
}

fn main() {
    let mut rng = rand::rng();

    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 250),
        FitnessEliminator::new_with_default(fitness),
        CrossoverRepopulator::new(0.25, CrossoverSettings::default()),
    );

    for i in 0..=150 {
        sim.next_generation();

        // sample a genome to print its fitness.
        // this value should approach 0 as the generations go on, since the fitness is negative error.
        // with the way CrossoverRepopulator (and all builtin repopulators) works internally, the parent genomes
        // (i.e. prev generation champs) are more likely to be at the start of the genomes vector.
        let sample = &sim.genomes[0];
        let fit = fitness(sample);
        println!("Gen {i} sample fitness: {fit}");
    }
    println!("Training complete, now you can test the network!");

    let net = &sim.genomes[0];
    println!("Network in use: {:#?}", net);

    loop {
        let mut input_text = String::new();
        println!("Enter a number to convert to degrees (or 'exit' to quit): ");
        std::io::stdin().read_line(&mut input_text).unwrap();
        let input_text = input_text.trim();
        if input_text.eq_ignore_ascii_case("exit") {
            break;
        }
        let input: f32 = match input_text.parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Invalid input, please enter a valid number.");
                continue;
            }
        };

        let output = net.predict([input])[0];
        let expected_output = input.to_degrees();
        println!(
            "Network output: {}, Expected output: {}, Error: {}",
            output,
            expected_output,
            (output - expected_output).abs()
        );
    }
}
