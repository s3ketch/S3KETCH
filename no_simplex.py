import contextlib
import time
import torch
import numpy as np
import pandas as pd
import contextlib
import time
import torch
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score

import numpy as np
from sklearn.model_selection import train_test_split
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

    return torch.sparse.mm(B1.permute(1, 0), B1) + torch.sparse.mm(B2, B2.permute(1, 0))


if __name__ == "__main__":
    datasets=['soc-sign-bitcoinalpha','soc-sign-bitcoinotc','Slashdot','epinions']
    for k in range(len(datasets)):
        data_name=datasets[k]
        data = pd.read_csv(f"./data/{data_name}.csv",header=None)
        if data_name in ['soc-sign-bitcoinalpha','soc-sign-bitcoinotc']:
            data.columns = ['SOURCE', 'TARGET', 'RATING','t']
        else:
            data.columns = ['SOURCE', 'TARGET', 'RATING'] 
        num_edges = len(data)
  
        edges_index = {}
        nodes = set(data['SOURCE']) | set(data['TARGET'])
        node_dict = {node: index for index, node in enumerate(nodes)}
        num_nodes = len(nodes)
    
        classes=[]
        graph = {}
      
        pos_in_degree = torch.zeros(num_nodes, 1, dtype=torch.float)
        neg_in_degree = torch.zeros(num_nodes, 1, dtype=torch.float)
        pos_out_degree = torch.zeros(num_nodes, 1, dtype=torch.float)
        neg_out_degree = torch.zeros(num_nodes, 1, dtype=torch.float)
        start=time.time()
        print(num_edges)
        edge_feature_tensor = torch.randn((num_edges, 2), dtype=torch.float).to(device)
        start=time.time()
        for index, row in data.iterrows():
            source = row['SOURCE']
            target = row['TARGET']
            rating = row['RATING']

            if rating > 0:
                category = 1
            else :
                category = 0
        
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
            if category == 1:
                pos_out_degree[source_index] += 1
                pos_in_degree[target_index] += 1
            else:
                neg_out_degree[source_index] += 1
                neg_in_degree[target_index] += 1
            classes.append(category)
        edge_index = torch.from_numpy(np.array(list(edges_index.keys())))
        triangles = set()
        triangle_num=0
        B2_row=[]
        B2_column=[]
        values=[]
    
        for u in range(num_nodes):
            neighbors_u = graph[u]
    
            for v in neighbors_u:
                if u < v: 
                    neighbors_v = graph[v]
                    common_neighbors = neighbors_u.intersection(neighbors_v)
                    for w in common_neighbors:
                        if u < v < w: 
                            
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
                        
                            B2_column.extend([triangle_num,triangle_num,triangle_num])
                            values.extend([1,-1,1])
                            triangle_num = triangle_num + 1
        B2_indices = torch.tensor([B2_row, B2_column], dtype=torch.long)
        B2_values = torch.tensor(values, dtype=torch.float)
        B2_shape = torch.Size([num_edges, triangle_num])
        B2= torch.sparse_coo_tensor(B2_indices, B2_values, B2_shape)        
        L=compute_Helmholtzians_Hodge_1_Laplacian(edge_index.T,B2,Nm=num_nodes,directed=True)
        indices = L._indices()
        values = L._values()

        values[values != 0] = 1

        L= torch.sparse_coo_tensor(indices, values, size=L.size())
        node_feature_tensor = torch.cat((pos_in_degree, neg_in_degree, pos_out_degree, neg_out_degree), dim=1)
        edge_feature_tensor = torch.zeros((num_edges, 4), dtype=torch.float)
        for i in range(num_edges):
            u, v = edge_index[i]
            edge_feature_tensor[i] = node_feature_tensor[u] + node_feature_tensor[v]

        T=[1,2,3,4,5]
        m=300
    
        pre_time = time.time()-start 
        
        
        start_time = time.time()
        for i in range(2):
            if i == 0:
                edge_feature_tensor_tmp = edge_feature_tensor.clone()
            
                

        
            W2 = torch.randn((edge_feature_tensor_tmp.size(1), m), dtype=torch.float)
        
            edge_feature_tensor_tmp = (torch.sparse.mm(L, torch.mm(edge_feature_tensor_tmp, W2))>0).float()
            
            embeds = edge_feature_tensor_tmp.numpy()
            negated_array = (1 - edge_feature_tensor_tmp).numpy()
                
            conta= np.hstack((edge_feature_tensor_tmp, negated_array))

        
            
              
        train_X, test_X, train_Y, test_Y = train_test_split(conta, classes, test_size=0.2)
        clf = LogisticRegression()
        clf.fit(train_X, train_Y)
        y_pred = clf.predict(test_X)
    
        f1 = f1_score(y_true=test_Y, y_pred=y_pred,average='binary')
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)
        print(data_name,f1,accuracy)
