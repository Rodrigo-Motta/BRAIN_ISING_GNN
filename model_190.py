import torch
import torch.nn.functional as func
from torch_geometric.nn import ChebConv, GCNConv, global_mean_pool, GATv2Conv, global_max_pool
from torch_geometric.nn import aggr


class GCN(torch.nn.Module):
    """GCN model(network architecture can be modified)"""

    def __init__(self,
                 num_features,
                 k_order,
                 dropout=.7):
        super(GCN, self).__init__()

        self.p = dropout

        # 190 ROIs
        # self.conv1 = ChebConv(int(num_features), 128, K=k_order)
        # self.conv2 = ChebConv(128, 128, K=k_order)
        # self.conv3 = ChebConv(128, 64, K=k_order)
        #
        # self.lin1 = torch.nn.Linear(64, 16)
        # self.lin2 = torch.nn.Linear(16, 1)

        # 333 ROIs
        self.conv1 = ChebConv(int(num_features), 64, K=k_order)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = ChebConv(64, 32, K=k_order)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.conv3 = ChebConv(32, 16, K=k_order)

        # self.bn3 = torch.nn.BatchNorm1d(32)

        self.lin1 = torch.nn.Linear(16, 4)
        self.lin2 = torch.nn.Linear(4, 1)

        torch.nn.init.xavier_normal_(self.lin1.weight)

        self.pool = global_mean_pool

        self.aggregation = aggr.SoftmaxAggregation()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        x = torch.nn.functional.leaky_relu(
            #self.bn1(
                self.conv1(x, edge_index))##, edge_attr))  # WHY NAN WITH EDGE_ATTR (non-negative)
        x = func.dropout(x, p=self.p, training=self.training)
        x = torch.nn.functional.leaky_relu(
            #self.bn2(
                self.conv2(x, edge_index))#, edge_attr))  # WHY NAN WITH EDGE_ATTR (non-negative)
        x = func.dropout(x, p=self.p, training=self.training)
        x = torch.nn.functional.leaky_relu(
            #self.bn3(
                self.conv3(x, edge_index))#, edge_attr))  # WHY NAN WITH EDGE_ATTR (non-negative)
        x = func.dropout(x, p=self.p, training=self.training)


        x = self.pool(x, batch)
        x1 = torch.nn.functional.relu(self.lin1(x))
        x2 = self.lin2(x1)
        return (x1, x2)