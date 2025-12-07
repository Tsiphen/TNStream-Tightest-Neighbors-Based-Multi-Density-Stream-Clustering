import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn import metrics
from tqdm import tqdm
import time
from collections import Counter, defaultdict
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import (
    normalized_mutual_info_score,
    f1_score,
    recall_score,
    accuracy_score,
    precision_score
)

# Annoy is the best choice for high-dimensional nearest neighbor search
try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False
    print("Warning: Annoy is not installed, and high-dimensional performance will be limited. Suggestion: pip install annoy")


class FastAnnoyIndex:
    def __init__(self, X, n_trees=10):
        self.X = np.asarray(X, dtype=np.float32)
        self.N, self.dim = self.X.shape
        self.n_trees = n_trees
        
        if ANNOY_AVAILABLE and self.N > 0:
            self.index = AnnoyIndex(self.dim, 'euclidean')
            for i in range(self.N):
                self.index.add_item(i, self.X[i])
            self.index.build(n_trees)
            self.use_annoy = True
        else:
            self.use_annoy = False
            # Degenerate to brute force search (acceptable for small data volumes)
    
    def query_knn(self, x, k=5):
        x = np.asarray(x, dtype=np.float32)
        
        if self.use_annoy:
            k = min(k, self.N)
            indices, distances = self.index.get_nns_by_vector(
                x, k, search_k=-1, include_distances=True
            )
            return list(zip(indices, distances))
        else:
            dists = np.linalg.norm(self.X - x, axis=1)
            idx = np.argsort(dists)[:k]
            return [(i, dists[i]) for i in idx]
    
    def query_radius(self, x, r, max_results=500):
        # Range query - Returns all points whose distance is less than r
        x = np.asarray(x, dtype=np.float32)
        
        if self.use_annoy:
            search_k = min(max_results, self.N)
            indices, distances = self.index.get_nns_by_vector(
                x, search_k, search_k=-1, include_distances=True
            )
            result = [idx for idx, dist in zip(indices, distances) if dist <= r]
            return result
        else:
            dists = np.linalg.norm(self.X - x, axis=1)
            return np.where(dists <= r)[0].tolist()
    
    def batch_query_knn(self, queries, k=5):
        results = []
        for q in queries:
            results.append(self.query_knn(q, k))
        return results


class AdaptiveParameters:
    # Adaptive parameter estimator
    
    @staticmethod
    def estimate_radius(X, sample_size=500, percentile=10):
        n = len(X)
        if n <= 1:
            return 1.0
        
        sample_size = min(sample_size, n)
        idx = np.random.choice(n, sample_size, replace=False)
        sample = X[idx]
        
        # Quickly calculate the distance between nearest neighbors
        if ANNOY_AVAILABLE and sample_size > 10:
            index = AnnoyIndex(X.shape[1], 'euclidean')
            for i, s in enumerate(sample):
                index.add_item(i, s.astype(np.float32))
            index.build(5)
            
            min_dists = []
            for i in range(sample_size):
                _, dists = index.get_nns_by_item(i, 6, include_distances=True)
                if len(dists) > 1:
                    min_dists.append(dists[1])
            
            if min_dists:
                return np.percentile(min_dists, percentile) * 1.5
        
        # Degradation scheme: Randomly sample and calculate the distance
        dists = []
        for i in range(min(100, sample_size)):
            for j in range(i + 1, min(i + 10, sample_size)):
                dists.append(np.linalg.norm(sample[i] - sample[j]))
        
        if dists:
            return np.percentile(dists, percentile)
        return 1.0
    
    @staticmethod
    def estimate_mk(dim):
        if dim <= 10:
            return 3
        elif dim <= 30:
            return 2
        else:
            return 1


class TNStreamOptimized:
    # LSH-TNStream High-performance Optimized Version
    
    def __init__(self, X, labels_true, N, W, r, n_micro, d, plotFigure, k, mk,
                 auto_params=True, verbose=True):
        self.X = X
        self.labels_true = labels_true
        self.k = k
        self.N = N
        self.W = W
        self.n_micro = n_micro
        self.plotFigure = plotFigure
        self.d = d
        self.verbose = verbose
        
        if auto_params:
            estimated_r = AdaptiveParameters.estimate_radius(X[:min(1000, len(X))])
            estimated_mk = AdaptiveParameters.estimate_mk(d)
            
            if verbose:
                print(f"Original parameters: r={r:.4f}, mk={mk}")
                print(f"Estimated parameters: r={estimated_r:.4f}, mk={estimated_mk}")
            
            if r < estimated_r * 0.5:
                print(f"Warning: Original r={r} is too small, Automatically adjust to {estimated_r:.4f}")
                r = estimated_r
            if mk > estimated_mk + 1:
                print(f"Warning: Original mk={mk} is too small, Automatically adjust to {estimated_mk}")
                mk = estimated_mk
        
        self.r = r
        self.mk = mk
        
        # State variable
        self.MC_Num = 0
        self.MacroC_Num = 0
        self.buffered_data = np.empty((0, d + 3), float)
        self.MCs = np.empty((0, d + 3), float)
        self.MCr = []
        self.MacroClusters = np.empty((0, 4), dtype=object)
        self.deleted_data = np.empty((0, d + 3), float)
        self.colors = np.empty((0, 4), int)
        
        # Index cache
        self._cached_index = None
        self._cached_data_hash = None
        
        # Performance statistics
        self.result = [[0, 0]]
        self.flag = 1
        self.wsize = 100
        
        self._run()
    
    def _get_data_hash(self, data):
        if len(data) == 0:
            return 0
        return hash((len(data), data[0, 0] if len(data) > 0 else 0))
    
    def _get_or_build_index(self, data, force_rebuild=False):
        if len(data) == 0:
            return None
        
        data_hash = self._get_data_hash(data)
        
        if (not force_rebuild and 
            self._cached_index is not None and 
            self._cached_data_hash == data_hash):
            return self._cached_index
        
        # Rebuild the index
        self._cached_index = FastAnnoyIndex(data, n_trees=8)
        self._cached_data_hash = data_hash
        return self._cached_index
    
    def _run(self):
        start = time.time()
        n_samples = len(self.X)
        
        for i in tqdm(range(n_samples), disable=not self.verbose):
            self.AddNode(self.X[i, :])
            self.DefineMC()
            self.AddtoMC()
            self.DefineMacroC()
            self.AddMCtoMacroC()
            self.UpdateInfo()
            self.UpdateMacroCs()
            self.KillMCs()
            self.KillMacroC()
            
            if i % self.wsize == 0 and i != 0:
                midd = time.time()
                elapsed = midd - start
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (n_samples - i) / rate if rate > 0 else 0
                
                self.result.append([i, elapsed])
                
                if self.verbose:
                    print(f"---> i={i}, Rate={rate:.1f} points per second, ≈{remaining/60:.1f} minutes remaining")
                
                # Timeout check
                if remaining > 3 * 3600:  # It is expected to take more than 3 hours
                    print("====Expected timeout, early termination====")
                    self.flag = 0
                    break
    
    def AddNode(self, X):
        if self.buffered_data.shape[0] == 0:
            new_row = np.hstack((np.array([1, 0, 0]), X))
        else:
            new_row = np.hstack((np.array([self.buffered_data[-1, 0] + 1, 0, 0]), X))
        
        self.buffered_data = np.vstack((self.buffered_data, new_row))
        
        # Sliding window
        if self.buffered_data.shape[0] > self.W:
            split = self.buffered_data.shape[0] - self.W
            self.deleted_data = np.vstack((self.deleted_data, self.buffered_data[:split]))
            self.buffered_data = self.buffered_data[split:]
            self._cached_index = None
    
    def DefineMC(self):
        unassigned_mask = self.buffered_data[:, 1] == 0
        X = self.buffered_data[unassigned_mask, :]
        tot = X.shape[0]
        
        if tot < self.N:
            return
        
        datatmp = X[:, 3:]
        index = FastAnnoyIndex(datatmp, n_trees=8)
        
        # Traverse to search for micro-cluster seeds
        for i in range(tot):
            # Range query
            knn_indx = index.query_radius(datatmp[i], r=self.r)
            if len(knn_indx) < self.N:
                continue
            maxr = 0
            valid_count = 0
            
            for j in knn_indx:
                if j >= tot:
                    continue
                
                knn_indy = index.query_radius(datatmp[j], r=self.r)
                snn = len(set(knn_indx) & set(knn_indy))
                
                if snn >= self.mk:
                    valid_count += 1
                    dist = np.linalg.norm(datatmp[j] - datatmp[i])
                    maxr = max(maxr, dist)
            
            if valid_count < self.N:
                continue
            
            maxr = min(maxr, self.r)
            
            # Points within the final range
            ind1 = index.query_radius(datatmp[i], r=maxr)
            points = [datatmp[l] for l in ind1 if l < tot]
            
            if len(points) >= self.N:
                center = np.mean(points, axis=0)
                self.MC_Num += 1
                self.MCr.append(maxr)
                self.MCs = np.vstack((
                    self.MCs,
                    np.hstack((np.array([self.MC_Num, len(points), 0]), center))
                ))
                
                # Assign points to micro-clusters
                for pt in points:
                    mask = np.all(self.buffered_data[:, 3:] == pt, axis=1)
                    self.buffered_data[mask, 1] = self.MC_Num
                
                return
    
    def AddtoMC(self):
        # Assign outliers to micro-clusters
        if self.MCs.shape[0] <= 1:
            return
        
        # Build a micro-cluster center index
        mc_centers = self.MCs[:, 3:]
        mc_index = FastAnnoyIndex(mc_centers, n_trees=5)
        
        # Batch process unallocated points
        unassigned_mask = self.buffered_data[:, 1] == 0
        unassigned_indices = np.where(unassigned_mask)[0]
        
        for i in unassigned_indices:
            result = mc_index.query_knn(self.buffered_data[i, 3:], k=1)
            if result:
                ind, d = result[0]
                if d <= self.MCr[ind]:
                    self.buffered_data[i, 1] = self.MCs[ind, 0]
    
    def UpdateInfo(self):
        for i in range(self.MCs.shape[0]):
            mask = self.buffered_data[:, 1] == self.MCs[i, 0]
            count = np.sum(mask)
            self.MCs[i, 1] = count
            
            if count > 0:
                self.MCs[i, 3:] = np.mean(self.buffered_data[mask, 3:], axis=0)
            
            if self.MCs[i, 2] >= 0:
                self.buffered_data[mask, 2] = self.MCs[i, 2]
    
    def KillMCs(self):
        for i in range(self.MCs.shape[0]):
            if int(self.MCs[i, 1]) == 0:
                mc_id = self.MCs[i, 0]
                macro_id = self.MCs[i, 2]
                
                self.buffered_data[self.buffered_data[:, 1] == mc_id, 1] = 0
                self.MCs = np.delete(self.MCs, i, axis=0)
                if i < len(self.MCr):
                    self.MCr.pop(i)
                
                if int(macro_id) != 0:
                    self.UpdateMacroC(macro_id)
                
                return
    
    def UpdateMacroCs(self):
        for i in range(self.MacroClusters.shape[0]):
            self.UpdateMacroC(self.MacroClusters[i, 0])
            self.MacroClusters[i, 1] = len(np.unique(self.MacroClusters[i, 2]))
    
    def UpdateMacroC(self, macroCluster):
        if macroCluster == 0:
            return
        
        P = self.MCs[self.MCs[:, 2] == macroCluster, :]
        if P.shape[0] == 0:
            return
        
        self.MCs[self.MCs[:, 2] == macroCluster, 2] = 0
        
        TN = self.search_TN(P[:, 3:], self.k)
        components = self.clustering(P[:, 3:], self.k, TN)
        
        if len(components) > 0:
            largest = max(components, key=len)
            mc_ids = P[list(largest), 0].astype(int)
            
            self.MacroClusters[self.MacroClusters[:, 0] == macroCluster, 2] = [mc_ids]
            self.MacroClusters[self.MacroClusters[:, 0] == macroCluster, 1] = len(mc_ids)
            
            for x in list(largest):
                j = int(P[x, 0])
                self.MCs[self.MCs[:, 0] == j, 2] = macroCluster
    
    def DefineMacroC(self):
        unassigned = self.MCs[self.MCs[:, 2] == 0, :]
        
        if len(unassigned) < self.n_micro:
            return
        
        mc_centers = unassigned[:, 3:]
        TN = self.search_TN(mc_centers, self.k)
        components = self.clustering(mc_centers, self.k, TN)
        
        for comp in components:
            if len(comp) >= self.n_micro:
                self.MacroC_Num += 1
                self.colors = np.array([
                    plt.cm.Spectral(x) for x in np.linspace(0, 1, self.MacroC_Num + 1)
                ])
                color = self.colors[-2, :]
                
                mc_ids = unassigned[list(comp), 0].astype(int)
                
                self.MacroClusters = np.vstack((
                    self.MacroClusters,
                    np.array([self.MacroC_Num, len(mc_ids), mc_ids, color], dtype=object)
                ))
                
                if self.verbose:
                    print(f"----------Macro Cluster #{self.MacroC_Num} defined----------")
                
                for mc_id in mc_ids:
                    self.MCs[self.MCs[:, 0] == mc_id, 2] = self.MacroC_Num
    
    def search_TN(self, X, k):
        # Search for TN-graph
        N = X.shape[0]
        
        if N <= 1:
            return [[[] for _ in range(k)] for _ in range(N)]
        
        # Batch query k-nearest neighbors using Annoy
        index = FastAnnoyIndex(X, n_trees=5)
        
        XIndex = []
        for i in range(N):
            result = index.query_knn(X[i], k=k + 1)
            indices = [r[0] for r in result]
            XIndex.append(indices)
        
        # Build TN
        KNN = [[] for _ in range(N)]
        RKNN = [[] for _ in range(N)]
        TN = [[[] for _ in range(k)] for _ in range(N)]
        
        for r in range(k):
            for j in range(N):
                if r + 1 < len(XIndex[j]):
                    y = XIndex[j][r + 1]
                    if y < N:
                        KNN[j].append(y)
                        RKNN[y].append(j)
            
            for i in range(N):
                if KNN[i] and RKNN[i]:
                    TN[i][r] = list(set(KNN[i]) & set(RKNN[i]))
        
        return TN
    
    def clustering(self, X, k, TN):
        # Clustering based on TN-graph
        N = X.shape[0]
        edges = []
        
        for i in range(N):
            if TN[i] and len(TN[i]) > k - 1 and TN[i][k - 1]:
                for j in TN[i][k - 1]:
                    if j > i:
                        edges.append((i, j))
        
        if not edges:
            return [set(range(N))]
        
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from(edges)
        
        return list(nx.connected_components(G))
    
    def AddMCtoMacroC(self):
        # Assign the micro-clusters to the macro clusters
        if self.MacroClusters.shape[0] == 0:
            return
        
        assigned = self.MCs[self.MCs[:, 2] != 0, :]
        if assigned.shape[0] == 0:
            return
        
        index = FastAnnoyIndex(assigned[:, 3:], n_trees=5)
        
        for i in range(self.MCs.shape[0]):
            if self.MCs[i, 2] == 0 and self.MCs[i, 1] >= self.N:
                result = index.query_knn(self.MCs[i, 3:], k=1)
                if result:
                    ind, _ = result[0]
                    self.MCs[i, 2] = assigned[ind, 2]
                    self.UpdateMacroC(self.MCs[i, 2])
                    return
    
    def KillMacroC(self):
        # Delete the unqualified macro clusters
        for i in range(self.MacroClusters.shape[0]):
            edges = self.MacroClusters[i, 2]
            total = sum(self.MCs[self.MCs[:, 0] == e, 1].sum() for e in edges)
            
            if total < self.n_micro * self.N and len(edges) < self.n_micro:
                self.MCs[self.MCs[:, 2] == self.MacroClusters[i, 0], 2] = 0
                if self.verbose:
                    print(f"----------Macro Cluster #{self.MacroClusters[i, 0]} killed----------")
                self.MacroClusters = np.delete(self.MacroClusters, i, axis=0)
                return
    
    def purity_score(self, y_true, y_pred):
        # Calculate Purity
        cm = metrics.cluster.contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(cm, axis=0)) / np.sum(cm)
    
    def compute_clustering_metrics(self, y_true, y_pred):
        # Calculate the clustering index
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = len(y_true)
        
        ari = adjusted_rand_score(y_true, y_pred)
        
        # Purity
        clusters = defaultdict(list)
        for idx, c in enumerate(y_pred):
            clusters[c].append(idx)
        purity = sum(
            max(Counter(y_true[idxs]).values())
            for idxs in clusters.values()
        ) / n
        
        nmi = normalized_mutual_info_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        return {
            "ARI": ari, "Purity": purity, "NMI": nmi,
            "F1_macro": f1, "Recall_macro": recall, "Accuracy": acc
        }
    
    def evalueate(self):
        # Evaluate the clustering results
        labels = np.hstack((self.deleted_data[:, 2], self.buffered_data[:, 2]))
        N = len(labels)
        t_labels = self.labels_true.reshape(-1)[:N]
        return self.compute_clustering_metrics(t_labels, labels)
    
    def plotGraph(self, title, dpi=70):
        # Draw 2D diagrams
        if self.d != 2:
            return
        
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        
        ax.scatter(self.buffered_data[:, 3], self.buffered_data[:, 4],
                  c='blue', alpha=0.15, s=20)
        
        for i, mc in enumerate(self.MCs):
            color = 'blue' if mc[2] == 0 else plt.cm.Spectral(mc[2] / max(1, self.MacroC_Num))
            circle = plt.Circle((mc[3], mc[4]), self.MCr[i],
                               color=color, fill=False, linewidth=1)
            ax.add_patch(circle)
            ax.plot(mc[3], mc[4], 'r.', markersize=3)
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(title)
        plt.show()
    
    def writetimetable(self, dataset_name):
        # Write to the schedule
        with open(f"{dataset_name}_LSH-TNtime.txt", "w") as f:
            for x in self.result:
                f.write(f"{x[0]}\t{x[1]:.2f}\n")
