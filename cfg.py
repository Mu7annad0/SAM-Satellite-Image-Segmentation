import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--net", default="sam", type=str)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--train_root", type=str,
                        default="../Dataset/DeepGlobeRoadExtraction/road/train/")
    parser.add_argument("--valid_root", type=str,
                        default="../Dataset/DeepGlobeRoadExtraction/road/valid/")
    parser.add_argument("--test_root", type=str,
                        default="../Dataset/DeepGlobeRoadExtraction/road/test/")
    parser.add_argument("--xd_train_root", type=str, default="../Dataset/XD/train/")
    parser.add_argument("--xd_valid_root", type=str, default="../Dataset/XD/valid/")
    parser.add_argument("--aerial_train_root", type=str, default="../Dataset/aerial/train/")
    parser.add_argument("--aerial_valid_root", type=str, default="../Dataset/aerial/valid/")
    parser.add_argument("--aerial_test_root", type=str, default="../Dataset/aerial/test/")
    parser.add_argument("--dubai_valid_root", type=str, default="../Dataset/Semantic segmentation dataset/valid/*/")
    parser.add_argument('--device', type=str, default="mps")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_type', type=str, default='vit_b', help='sam model_type')
    parser.add_argument('--work_dir', type=str, default='../workdir/')
    parser.add_argument('--checkpoint', type=str, default='../workdir/ck/SAM_checkpoint/sam_vit_b_01ec64.pth')
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--run_name', type=str, default='sam-satellite-models', help="run model name")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--propmt_grad', type=bool, default=True, help="fine-tune the propmt encoder")
    parser.add_argument('--num_points', type=int, default=10)
    parser.add_argument('--box', type=bool, default=True, help="using box as prompt")
    parser.add_argument('--point', type=bool, default=True, help="using points as prompt")
    parser.add_argument('--use_adapter', type=bool, default=False, help="using adpter in the training")
    parser.add_argument("--point_list", type=list, default=[5, 10, 15, 20], help="point_list")
    parser.add_argument("--point_iterator", type=list, default=20, help="points iteration")
    parser.add_argument("--use_scheduler", type=bool, default=True, help="use scheduler")
    parser.add_argument("--early_stop", type=bool, default=True, help="early stop the training process")
    parser.add_argument("--seed", type=int, default=2049, help="random seed factor")
    parser.add_argument("--resume_training", type=str, default=None, help="resume training") 
    parser.add_argument("--multimask", type=bool, default=False, help="ouput multimask")
    args = parser.parse_args()
    return args
