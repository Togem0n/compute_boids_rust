pub mod lib;

fn main() {
    pollster::block_on(lib::run());
}