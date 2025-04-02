use neat::*;
use rand::prelude::*;

// derive some traits so that we can use this agent with `genetic-rs`.
#[derive(Debug, Clone, PartialEq, CrossoverReproduction, DivisionReproduction, RandomlyMutable)]
struct MyAgentGenome {
    brain: NeuralNetwork<3, 2>,
}

impl Prunable for MyAgentGenome {}

impl GenerateRandom for MyAgentGenome {
    // allows us to use `Vec::gen_random` for the initial population.
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self {
            brain: NeuralNetwork::new(MutationSettings::default(), rng),
        }
    }
}

// creates a bigger number within the bounds of 0 to 1 as `actual` approaches `expected`.
fn inverse_error(expected: f32, actual: f32) -> f32 {
    1.0 / (1.0 + (expected - actual).abs())
}

fn fitness(agent: &MyAgentGenome) -> f32 {
    let mut rng = rand::thread_rng();
    let mut fit = 0.;

    for _ in 0..10 {
        // run the test multiple times for consistency

        let inputs = [rng.gen(), rng.gen(), rng.gen()];

        // try to force the network to learn to do some basic logic
        let expected0: f32 = (inputs[0] >= 0.5 && inputs[1] < 0.5).into();
        let expected1: f32 = (inputs[2] >= 0.5).into();

        // println!("predicting {i}");
        let output = agent.brain.predict(inputs);

        fit += inverse_error(expected0, output[0]);
        fit += inverse_error(expected1, output[1]);
    }

    fit
}

fn main() {
    let mut sim = GeneticSim::new(
        // create a population of 100 random neural networks
        Vec::gen_random(2),
        // provide the fitness function that will
        // test the agents individually so the nextgen
        // function can eliminate the weaker ones.
        fitness,
        // this nextgen function will kill/drop agents
        // that don't have a high enough fitness, and repopulate
        // by performing crossover reproduction between the remaining ones
        division_pruning_nextgen,
    );

    // fast forward 100 generations. identical to looping
    // 100 times with `sim.next_generation()`.
    for i in 0..100000 {
        println!("{i}");
        sim.next_generation();
    }
}
