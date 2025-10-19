'''

creat by kun at Oct 2021
Reference: https://github.com/xiaxin1998/DHCN
'''



import argparse
import pickle
import time
from util import Data, split_validation
from model import *
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='electronics', help='dataset name: amazon/digineticaBuy/cosmetics/electronics')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
parser.add_argument('--embSize', type=int, default=128, help='embedding size')
parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=1, help='the number of layer used, 1 for electronics, 2 for Multi-category, 3 for Cosmetics')
parser.add_argument('--lambda1', type=float, default=0.1, help='contrastive loss')
parser.add_argument('--lambda2', type=float, default=0.1, help='price trend')
parser.add_argument('--beta', type=float, default=0.1, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

opt = parser.parse_args()
print(opt)

torch.cuda.set_device(0)

def main():
    # list[0]:session list[1]:label
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))

    if opt.dataset == 'cosmetics':
        n_node = 23194
        n_user = 93665
        n_price = 10
        n_category = 301
    elif opt.dataset == 'electronics':
        n_node = 1179
        n_user = 13158
        n_price = 20
        n_category = 141
    elif opt.dataset == 'multi-category':
        n_node = 5221
        n_user = 212291
        n_price = 50
        n_category = 260
    else:
        print("unkonwn dataset")
    # data_formate: sessions, price_seq, matrix_session_item, matrix_session_price, matrix_pv, matrix_pb, matrix_pc, matrix_bv, matrix_bc, matrix_cv
    train_data = Data(train_data, shuffle=True, n_node=n_node, n_user=n_user, n_price=n_price, n_category=n_category)
    test_data = Data(test_data, shuffle=True, n_node=n_node, n_user=n_user, n_price=n_price, n_category=n_category)
    model = trans_to_cuda(PGCA(adjacency=train_data.adjacency,
                                adjacency_pv=train_data.adjacency_pv,
                                adjacency_vp=train_data.adjacency_vp,
                                adjacency_uv=train_data.adjacency_uv,
                                adjacency_vu=train_data.adjacency_vu,
                                adjacency_pc=train_data.adjacency_pc,
                                adjacency_cp=train_data.adjacency_cp,
                                adjacency_cv=train_data.adjacency_cv,
                                adjacency_vc=train_data.adjacency_vc,
                                n_node=n_node,
                                n_user=n_user,
                                n_price=n_price,
                                n_category=n_category,
                                lr=opt.lr, l2=opt.l2, beta=opt.beta,
                                layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,
                                dataset=opt.dataset, num_heads=opt.num_heads, lambda1=opt.lambda1, lambda2=opt.lambda2))

    top_K = [1, 5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    for epoch in range(1, opt.epoch + 1):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data, opt.lambda1)
        # 标记是否有任何更新
        model_updated = False

        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100

            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                model_updated = True

            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
                model_updated = True

            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][2] = epoch
                model_updated = True
        print('P@1\tP@5\tM@5\tN@5\tP@10\tM@10\tN@10\tP@20\tM@20\tN@20\t')
        print("%.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f" % (
            best_results['metric1'][0], best_results['metric5'][0], best_results['metric5'][1],
            best_results['metric5'][2], best_results['metric10'][0], best_results['metric10'][1],
            best_results['metric10'][2], best_results['metric20'][0], best_results['metric20'][1],
            best_results['metric20'][2]))
        print("%d\t %d\t %d\t %d\t %d\t %d\t % d\t %d\t %d\t %d" % (
            best_results['epoch1'][0], best_results['epoch5'][0], best_results['epoch5'][1],
            best_results['epoch5'][2], best_results['epoch10'][0], best_results['epoch10'][1],
            best_results['epoch10'][2], best_results['epoch20'][0], best_results['epoch20'][1],
            best_results['epoch20'][2]))

        # 只有在模型更新时才保存模型
        if model_updated:
            # 在训练脚本（main.py）中添加
            if not os.path.exists('./saved_models/' + opt.dataset):
                os.makedirs('./saved_models/' + opt.dataset)

            # 保存最佳模型（结构和参数）
            best_model_path = './saved_models/' + opt.dataset + '/best_model_full.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'metrics': metrics,
            }, best_model_path)
            print('模型已更新，当前最好次数为：', epoch)

    # 训练完成后进行可视化
    print('开始可视化特征分布对比...')

    # 确保有保存模型的目录
    viz_dir = './visualizations/' + opt.dataset
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    # 加载最佳模型进行可视化
    best_model_path = './saved_models/' + opt.dataset + '/best_model_full111.pth'
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'已加载最佳模型 (epoch {checkpoint["epoch"]})')

        # 进行可视化
        viz_results = visualize_fusion_comparison(
            model,
            test_data,
            sample_batch=0,  # 使用第一个batch
            save_path=f'{viz_dir}/feature_fusion_comparison.pdf'
        )

        # 打印详细统计信息
        print("\n=== 特征融合详细统计 ===")
        stats = viz_results['change_stats']
        print(f"平均变化幅度: {stats['mean_change']:.4f}")
        print(f"最大变化幅度: {stats['max_change']:.4f}")
        print(f"平均变化比例: {stats['change_ratio']:.4f}")

    else:
        print("未找到保存的最佳模型，使用当前模型进行可视化")
        # 使用当前模型进行可视化
        visualize_fusion_comparison(
            model,
            test_data,
            sample_batch=0,
            save_path=f'{viz_dir}/feature_fusion_comparison.pdf'
        )

if __name__ == '__main__':
    main()
