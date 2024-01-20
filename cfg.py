import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--train_root", type=str, default="../../graduating_project/Dataset/DeepGlobeRoadExtraction/road/train/")
    parser.add_argument("--valid_root", type=str, default="../../graduating_project/Dataset/DeepGlobeRoadExtraction/road/valid/")
    parser.add_argument("--test_root", type=str, default="../../graduating_project/Dataset/DeepGlobeRoadExtraction/road/test/")
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_type', type=str, default='vit_b', help='sam model_type')
    parser.add_argument('--work_dir', type=str, default='../workdir/')
    parser.add_argument('--checkpoint', type=str, default='../wordir/SAM/')
    parser.add_argument('--run_name', type=str, default='sam-satellite', help="run model name")
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')

    args = parser.parse_args()
    return args 