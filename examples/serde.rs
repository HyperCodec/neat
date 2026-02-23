use neat::{activation::register_activation, *};

const OUTPUT_PATH: &str = "network.json";

fn magic_activation(x: f32) -> f32 {
    // just a random activation function to show that it gets serialized and deserialized correctly.
    (x * 2.0).sin()
}

fn main() {
    // custom activation functions must be registered before deserialization, since the network needs to know how to deserialize them.
    register_activation(activation_fn!(magic_activation));

    let mut rng = rand::rng();
    let mut net = NeuralNetwork::<10, 10>::new(&mut rng);

    println!("Mutating network...");

    for _ in 0..100 {
        net.mutate(&MutationSettings::default(), 0.25, &mut rng);
    }

    let file =
        std::fs::File::create(OUTPUT_PATH).expect("Failed to create file for network output");
    serde_json::to_writer_pretty(file, &net).expect("Failed to write network to file");

    println!("Network saved to {OUTPUT_PATH}");

    // reopen because for some reason io hates working properly with both read and write
    // (even when using OpenOptions)
    let file = std::fs::File::open(OUTPUT_PATH).expect("Failed to open network file for reading");
    let net2: NeuralNetwork<10, 10> =
        serde_json::from_reader(file).expect("Failed to parse network from file");
    assert_eq!(net, net2);
    println!("Network successfully loaded from file and matches original!");
}
