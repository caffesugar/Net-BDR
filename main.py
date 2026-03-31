import argparse
from engine import PhaseRetrievalEngine

def parse_args():
    parser = argparse.ArgumentParser(description="Net-BDR for Fourier Phase Retrieval")
    
    # Core experimental variables (CLI accessible)
    parser.add_argument('--k', type=float, default=0.2, help='Padding ratio for background information')
    parser.add_argument('--noise_std', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0)')
    parser.add_argument('--max_iter', type=int, default=300, help='Maximum number of outer iterations')
    parser.add_argument('--runs', type=int, default=5, help='Number of independent runs with fixed random seeds')
    parser.add_argument('--is_net', action='store_true', default=True, help='Enable network prior optimization')
    
    # Path configurations
    parser.add_argument('--input_dir', type=str, default='./test_images', help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save outputs and metrics')
    
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use (default: 0)')
    
    return parser.parse_args()

def main():
    # 1. Parse command-line arguments
    args = parse_args()
    
    # 2. Initialize the experimental engine
    engine = PhaseRetrievalEngine(args)
    
    # 3. Execute the batch evaluation pipeline
    engine.run()

if __name__ == '__main__':
    main()