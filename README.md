# Theoretical Part of the Project

## Introduction

This is the theoretical part of my project, Graph-Based agent system. This repo only include the modules I created or modified. To run the experiment, the GPTSwarm framework should be installed first. Please refer to https://github.com/metauto-ai/GPTSwarm for more details.

It includes the following sections:

1. I propose a new method in edge optimization with PPO, which is included in the evaluator.optimize_swarm_ppo and swarm.optimizer.edge_optimizer.parameterization.PPOEdgeWiseDistribution.

    optimize_swarm_ppo module:
    ```python
        async def optimize_swarm_ppo(
            self,
            num_iters: int,
            lr: float,
            batch_size: int = 4,
        ) -> torch.Tensor:
        
        assert self._swarm is not None
        
        dataset = self._train_dataset
        
        print(f"Optimizing swarm on {dataset.__class__.__name__} split {dataset.split}")
        optimizer = torch.optim.Adam(self._swarm.connection_dist.parameters(), lr=lr)
        
        if self._art_dir_name is not None:
            hp_json_name = os.path.join(self._art_dir_name, "hp.json")
            with open(hp_json_name, "w") as f:
                json.dump(dict(lr=lr,
                               batch_size=batch_size,
                               num_iters=num_iters,
                               model_name=self._model_name
                               ), f)
        
        def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record
        
        loader = infinite_data_loader()
        
        edge_probs = None
        old_log_probs = None
        for i_iter in range(num_iters):
            print(f"Iter {i_iter}", 80 * '-')
        
            start_ts = time.time()
        
            future_answers = []
            log_probs = []
            correct_answers = []
        
            for i_record, record in zip(range(batch_size), loader):
                realized_graph, log_prob = self._swarm.connection_dist.realize(
                    self._swarm.composite_graph,
                    # temperature=3.0, # DEBUG
                )
        
                input_dict = dataset.record_to_swarm_input(record)
                answer = self._swarm.arun(input_dict, realized_graph)
                future_answers.append(answer)
                log_probs.append(log_prob)
                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)
        
            raw_answers = await asyncio.gather(*future_answers)
        
            print(f"Batch time {time.time() - start_ts:.3f}")
        
            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
        
            # Initialize old_log_probs if this is the first iteration
            if old_log_probs is None:
                old_log_probs = log_probs
        
            baseline = 0.5
            for raw_answer, log_prob, correct_answer, old_log_prob in zip(raw_answers,
                                                                          log_probs,
                                                                          correct_answers,
                                                                          old_log_probs):
                answer = dataset.postprocess_answer(raw_answer)
                assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {correct_answer} (1)"
        
                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                utility = accuracy.get()
                utilities.append(utility)
        
                advantage = utility - baseline
        
                ratio = torch.exp(log_prob - old_log_prob)
        
                clip_value = 0.2
                clipped_ratio = torch.clamp(ratio, 1 - clip_value, 1 + clip_value)
        
                single_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
                loss_list.append(single_loss)
        
            old_log_probs = log_probs
        
            mean_utility = np.mean(np.array(utilities))
            total_loss = torch.mean(torch.stack(loss_list))
        
            print("utilities:", utilities)
            print("loss:", total_loss.item())
        
            # Perform the optimization step
            optimizer.zero_grad()
            self._swarm.connection_dist.update(optimizer, mean_utility)
            total_loss.backward(retain_graph=True)
            print("Grad:", self._swarm.connection_dist.edge_logits.grad)
            optimizer.step()
        
            print("edge_logits:", self._swarm.connection_dist.edge_logits)
            edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
            print("edge_probs:", edge_probs)
        
            self._print_conns(edge_probs)
        
            if self._logger is not None:
                self._logger.add_scalar("train/loss", total_loss.item(), i_iter)
                self._logger.add_scalar("train/utility", mean_utility.item(), i_iter)
            if self._art_dir_name is not None:
                log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
                with open(log_jsonl_name, "a") as f:
                    json.dump(dict(iter=i_iter, train_loss=total_loss.item(), train_utility=mean_utility.item()), f)
                    f.write("\n")
            print("end of iteration")
        
        if edge_probs is not None:
            self._print_conns(edge_probs, save_to_file=True)
        
        print("Done!")
        edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
        return edge_probs
    ```
    
    PPOEdgeWiseDistribution module:
    ```python
    class PPOEdgeWiseDistribution(ConnectDistribution):
        def __init__(self,
                     potential_connections,
                     initial_probability: float = 0.5,
                     clip_param: float = 0.2,
                     ):
            super().__init__(potential_connections)
            init_logit = torch.log(torch.tensor(initial_probability / (1 - initial_probability)))
            init_tensor = torch.ones(
                len(potential_connections),
                requires_grad=True) * init_logit
            self.edge_logits = torch.nn.Parameter(init_tensor)
            self.clip_param = clip_param
            self.old_edge_logits = None
            node_ids = set([x for pair in potential_connections for x in pair])
            self.node_idx2id = {i: node_id for i, node_id in enumerate(node_ids)}
            self.node_id2idx = {node_id: i for i, node_id in enumerate(node_ids)}
            order_tensor = torch.randn(len(node_ids))
            self.order_params = torch.nn.Parameter(order_tensor)
    
            self.store_old_logits()
    
        def store_old_logits(self):
            self.old_edge_logits = self.edge_logits.clone().detach()
    
        def random_sample_num_edges(self, graph: CompositeGraph, num_edges: int) -> CompositeGraph:
            _graph = deepcopy(graph)
            while True:
                if _graph.num_edges >= num_edges:
                    break
                potential_connection = random.sample(self.potential_connections, 1)[0]
                out_node = _graph.find_node(potential_connection[0])
                in_node = _graph.find_node(potential_connection[1])
    
                if not out_node or not in_node:
                    continue
    
                if not _graph.check_cycle(in_node, {out_node}, set()):
                    out_node.add_successor(in_node)
                    in_node.add_predecessor(out_node)
            return _graph
    
        def realize_ranks(self, graph, use_max: bool = False):
            log_probs = []
            ranks = {}
            in_degrees = {node.id: len(node.predecessors) for node in graph.nodes.values()}
            for i in range(len(self.order_params)):
                available_nodes = [node for node in graph.nodes if in_degrees[node] == 0]
                logits = []
                for node in available_nodes:
                    logits.append(self.order_params[self.node_id2idx[node]])
                logits = torch.stack(logits).reshape(-1)
                if use_max:
                    idx = torch.argmax(logits)
                else:
                    idx = torch.distributions.Categorical(logits=logits).sample()
    
                log_probs.append(torch.log_softmax(logits, dim=0)[idx])
    
                ranks[available_nodes[idx]] = i
                in_degrees[available_nodes[idx]] = -1
                for successor in graph.nodes[available_nodes[idx]].successors:
                    in_degrees[successor.id] -= 1
            return ranks, torch.sum(torch.stack(log_probs))
    
        def realize(self,
                    graph: CompositeGraph,
                    temperature: float = 1.0,  # must be >= 1.0
                    threshold: float = 0.5,
                    use_learned_order: bool = False,
                    ) -> Tuple[CompositeGraph, torch.Tensor]:
            if use_learned_order:
                ranks, log_prob = self.realize_ranks(graph, threshold is not None)
                log_probs = [log_prob]
            else:
                log_probs = [torch.tensor(0.0, requires_grad=True)]
            _graph = deepcopy(graph)
            for potential_connection, edge_logit in zip(
                    self.potential_connections, self.edge_logits):
                out_node = _graph.find_node(potential_connection[0])
                in_node = _graph.find_node(potential_connection[1])
    
                if not out_node or not in_node:
                    continue
    
                addable_if_use_learned_order = use_learned_order and (ranks[out_node.id] < ranks[in_node.id])
                addable_if_not_used_learned_order = (not use_learned_order) and (
                    not _graph.check_cycle(in_node, {out_node}, set()))
                if addable_if_not_used_learned_order or addable_if_use_learned_order:
                    edge_prob = torch.sigmoid(edge_logit / temperature)
                    if threshold:
                        edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                    if torch.rand(1) < edge_prob:
                        out_node.add_successor(in_node)
                        log_probs.append(torch.log(edge_prob))
                    else:
                        log_probs.append(torch.log(1 - edge_prob))
    
            log_prob = torch.sum(torch.stack(log_probs))
            return _graph, log_prob
    
        def realize_full(self, graph: CompositeGraph) -> CompositeGraph:
            _graph = deepcopy(graph)
            for i, potential_connection in enumerate(self.potential_connections):
                out_node = _graph.find_node(potential_connection[0])
                in_node = _graph.find_node(potential_connection[1])
    
                if not out_node or not in_node:
                    continue
    
                if not _graph.check_cycle(in_node, {out_node}, set()):
                    out_node.add_successor(in_node)
                    in_node.add_predecessor(out_node)
            return _graph
    
        def realize_mask(self, graph: CompositeGraph, edge_mask: torch.Tensor) -> CompositeGraph:
            _graph = deepcopy(graph)
            for i, (potential_connection, is_edge) in enumerate(zip(self.potential_connections, edge_mask)):
                out_node = _graph.find_node(potential_connection[0])
                in_node = _graph.find_node(potential_connection[1])
    
                if not out_node or not in_node:
                    continue
    
                if not _graph.check_cycle(in_node, {out_node}, set()):
                    if is_edge:
                        out_node.add_successor(in_node)
                        in_node.add_predecessor(out_node)
            return _graph
    
        def ppo_loss(self, log_probs, rewards):
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards)
    
            old_log_probs = self.old_edge_logits
            ratios = torch.sum(torch.exp(log_probs - old_log_probs))
            clipped_ratios = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param)
    
            loss = -torch.min(ratios * rewards, clipped_ratios * rewards).mean()
            self.store_old_logits()
    
            return loss
    
        def update(self, optimizer, rewards):
            log_probs = torch.stack([self.edge_logits], dim=0)
            loss = self.ppo_loss(log_probs, rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    ```
   
2. I propose two new datasets, which is inside the datasets folder, called Math_Code and DifficultCodeGeneration.

3. I implement a plot_graphs module to plot the swarm of agents in my edge optimization experiment, which is called plot_graphs.py. To use it, simply run the following command:
    ```bash
    python plot_graphs.py --file_path <path_to_connections.txt>
    ```
   The path_to_connections.txt is the path to the connections.txt file generated by the edge optimization experiment.

