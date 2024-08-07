import contextlib
import time
import torch
import numpy as np
import pandas as pd
import time
from scipy.io import savemat
def compute_Helmholtzians_Hodge_1_Laplacian(edge_index, B2, Nm=None, directed=True):
 
    edge_index = edge_index
    print(edge_index.shape)
    if Nm is None:
        Nm = torch.max(edge_index) + 1

    B1_row=[]
    B1_column=[]
    values=[]
    for i in range(edge_index.shape[1]):
        B1_row.extend([edge_index[0, i],edge_index[1, i]])
        B1_column.extend([i,i])
        values.extend([-1,1])
    B1_indices = torch.tensor([B1_row, B1_column], dtype=torch.long)
    B1_values = torch.tensor(values, dtype=torch.float)
    B1_shape = torch.Size([Nm, edge_index.shape[1]])

    B1= torch.sparse_coo_tensor(B1_indices, B1_values, B1_shape)

    print(B1.shape, B2.shape) 


    return torch.sparse.mm(B1.permute(1, 0), B1) + torch.sparse.mm(B2, B2.permute(1, 0)),B1



if __name__ == "__main__":

    edges_list =[1e7, 3e7, 5e7, 7e7, 9e7]

   
    cost_result=np.zeros((5, ))
    f1_result=np.zeros((5, ))
    acc_result=np.zeros((5, ))
    for j in range(len(edges_list)):
        edge_num=int(edges_list[j])
        data = pd.read_csv(f'./data/Scalability/edges/edge_index_edge{edge_num}.csv',header=None)
        data.columns = ['SOURCE', 'TARGET']
        num_edges = len(data)
        edges_index = {}
        nodes = set(data['SOURCE']) | set(data['TARGET'])
        node_dict = {node: index for index, node in enumerate(nodes)}
        num_nodes = len(nodes)
        graph = {}
        pos_in_degree = torch.zeros(num_nodes, 1, dtype=torch.float)
        neg_in_degree = torch.zeros(num_nodes, 1, dtype=torch.float)
        pos_out_degree = torch.zeros(num_nodes, 1, dtype=torch.float)
        neg_out_degree = torch.zeros(num_nodes, 1, dtype=torch.float)
        start=time.time()
       
        edge_feature_tensor = torch.randn((num_edges, 2), dtype=torch.float)
        for index, row in data.iterrows():
            source = row['SOURCE']
            target = row['TARGET']
            source_index = node_dict[source]
            target_index = node_dict[target]
            if source_index not in graph:
                graph[source_index] = set()
            graph[source_index].add(target_index)
            if target_index not in graph:
                graph[target_index] = set()
            graph[target_index].add(source_index)
            if (source_index, target_index) in edges_index:
                print(f"Duplicate edge found: ({source_index}, {target_index})")
            else:
                edges_index[(source_index, target_index)] = index
                # edges_index[(source_index, target_index)] = index
            
        edge_index = torch.from_numpy(np.array(list(edges_index.keys())))  
        triangles = set()
        # triangles = {}
        triangle_num=0
        B2_row=[]
        B2_column=[]
        values=[]
    
        for u in range(num_nodes):
            neighbors_u = graph[u]
    
            for v in neighbors_u:
                if u < v:  # Ensure each triangle is listed once
                    neighbors_v = graph[v]
                    common_neighbors = neighbors_u.intersection(neighbors_v)
                    for w in common_neighbors:
                        if u < v < w:  # Ensure a consistent ordering
                            
                            triangles.add((u, v, w))
                            edge_index_uv = edges_index.get((u, v), None)
                            if edge_index_uv is None:
                                edge_index_uv = edges_index.get((v, u), None)
                                if edge_index_uv is None:
                                  
                                    continue  

                          
                            edge_index_uw = edges_index.get((u, w), None)
                            if edge_index_uw is None:
                                edge_index_uw = edges_index.get((w, u), None)
                                if edge_index_uw is None:
                                    continue  
                            edge_index_vw = edges_index.get((v, w), None)
                            if edge_index_vw is None:
                                edge_index_vw = edges_index.get((w, v), None)
                                if edge_index_vw is None:
                                    continue 

                            B2_row.extend([edge_index_uv, edge_index_uw, edge_index_vw])
                            # B2_row.append([edges_index[(u,v)],edges_index[(u,w)],edges_index[(v,w)]])
                            B2_column.extend([triangle_num,triangle_num,triangle_num])
                            values.extend([1,-1,1])
                            triangle_num = triangle_num + 1
        B2_indices = torch.tensor([B2_row, B2_column], dtype=torch.long)
        B2_values = torch.tensor(values, dtype=torch.float)
        B2_shape = torch.Size([num_edges, triangle_num])
        B2= torch.sparse_coo_tensor(B2_indices, B2_values, B2_shape)        
        L,B1=compute_Helmholtzians_Hodge_1_Laplacian(edge_index.T,B2,Nm=num_nodes,directed=True)
 
        node_feature_tensor = torch.randn(num_nodes, 4)
        edge_feature_tensor = torch.randn(num_edges, 4)
        triange_feature_tensor = torch.randn(len(triangles), 4)

        
        T=[1,2,3,4,5]
        m=300
        t=3
        for i in range(t):
            if i == 0:
                edge_feature_tensor_tmp = edge_feature_tensor.clone()
                node_feature_tensor_tmp = node_feature_tensor.clone()
                triange_feature_tensor_tmp = triange_feature_tensor.clone()

          
            W1 = torch.randn((node_feature_tensor_tmp.size(1), m), dtype=torch.float)
            W2 = torch.randn((edge_feature_tensor_tmp.size(1), m), dtype=torch.float)
            W3 = torch.randn((triange_feature_tensor_tmp.size(1), m), dtype=torch.float)

            node2edge = torch.sparse.mm(B1.t(), torch.mm(node_feature_tensor_tmp, W1))
            edge_feature_tensor_tmp = torch.sparse.mm(L, torch.mm(edge_feature_tensor_tmp, W2))
            triangle2edge = torch.mm(B2, torch.mm(triange_feature_tensor_tmp, W3))

            node2edge = torch.round(torch.sigmoid(node2edge))
            edge_feature_tensor_tmp = torch.round(torch.sigmoid(edge_feature_tensor_tmp))
            triangle2edge = torch.round(torch.sigmoid(triangle2edge))
            node_feature_tensor_tmp = torch.round(torch.sigmoid(node_feature_tensor_tmp))
            triange_feature_tensor_tmp = torch.round(torch.sigmoid(triange_feature_tensor_tmp))

            edge_feature_tensor_tmp = node2edge + edge_feature_tensor_tmp + triangle2edge
            embeds = edge_feature_tensor_tmp.numpy()


        cost_result[j] = time.time() - start
        
            # 保存结果
    print(cost_result)
    save_path = f'./result/scal.mat'
    sio.savemat(save_path, {'cost': cost_result})