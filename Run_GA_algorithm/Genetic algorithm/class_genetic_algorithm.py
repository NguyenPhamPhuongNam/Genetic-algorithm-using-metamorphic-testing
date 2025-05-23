# --- 9) Lớp GeneticAlgorithm ---
class GeneticAlgorithm:
    def __init__(self, img_shape, seg_model, compute_iou,
                 pop_size=10, generations=5,
                 crossover_method="two_point",
                 add_del_rate=0.2,
                 param_mut_rate=0.1,
                 alpha=1.0, beta=1.0):
        self.img_shape = img_shape
        self.seg_model = seg_model
        self.compute_iou = compute_iou
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_method = crossover_method
        self.add_del_rate = add_del_rate
        self.param_mut_rate = param_mut_rate
        self.alpha = alpha
        self.beta = beta
        self.population = [Chromosome(img_shape) for _ in range(pop_size)]

    def select(self):
        self.population.sort(key=lambda c: c.fitness if c.fitness is not None else -np.inf, reverse=True)
        self.population = self.population[:self.pop_size // 2] * 2

    def crossover(self):
        next_gen = []
        for i in range(0, self.pop_size, 2):
            p1, p2 = self.population[i], self.population[i + 1]
            t1, t2 = p1.transformations, p2.transformations
            L = max(len(t1), len(t2))
            pad1 = [generate_random_subvector(self.img_shape)] * (L - len(t1))
            pad2 = [generate_random_subvector(self.img_shape)] * (L - len(t2))
            t1 = t1 + pad1
            t2 = t2 + pad2

            if L >= 3:
                a, b = sorted(random.sample(range(1, L), 2))
            elif L == 2:
                a, b = 1, 2
            else:
                a, b = 0, 0

            c1 = t1[:a] + t2[a:b] + t1[b:]
            c2 = t2[:a] + t1[a:b] + t2[b:]
            next_gen += [
                Chromosome(self.img_shape, transformations=copy.deepcopy(c1)),
                Chromosome(self.img_shape, transformations=copy.deepcopy(c2))
            ]
        self.population = next_gen

    def mutate(self):
        for chromo in self.population:
            if random.random() < self.add_del_rate and len(chromo.transformations) > 1:
                chromo.transformations.pop(random.randrange(len(chromo.transformations)))
            if random.random() < self.add_del_rate:
                pos = random.randrange(len(chromo.transformations) + 1)
                chromo.transformations.insert(pos, generate_random_subvector(self.img_shape))
            for sub in chromo.transformations:
                if random.random() < self.param_mut_rate:
                    sub["activate_dropout"] ^= 1
                    sub["dropout_rate"] = np.clip(sub["dropout_rate"] + random.uniform(-0.01, 0.01), 0.01, 0.1)
                if random.random() < self.param_mut_rate:
                    sub["activate_gaussian"] ^= 1
                    sub["gaussian_sigma"] = np.clip(sub["gaussian_sigma"] + random.uniform(-0.5, 0.5), 1, 5)
                if random.random() < self.param_mut_rate:
                    sub["activate_brightness"] ^= 1
                    sub["brightness_shift"] = np.clip(sub["brightness_shift"] + random.uniform(-0.05, 0.05), -0.2, 0.2)
                if random.random() < self.param_mut_rate:
                    sub["activate_channel_shift"] ^= 1
                    i = random.randrange(3)
                    sub["channel_shift_values"][i] = np.clip(
                        sub["channel_shift_values"][i] + random.uniform(-0.05, 0.05), -0.2, 0.2
                    )

    def run(self, img, true_mask):
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = img.copy()
        if isinstance(true_mask, torch.Tensor):
            mask_np = true_mask.cpu().numpy()
        else:
            mask_np = true_mask.copy()

        for gen in range(self.generations):
            for c in self.population:
                evaluate_fitness(img_np, mask_np, c, self.seg_model, alpha=self.alpha, beta=self.beta)
            best = max(self.population, key=lambda c: c.fitness if c.fitness is not None else -np.inf)
            print(f"Gen {gen}: best fitness = {best.fitness:.4f}, length = {len(best.transformations)}")
            self.select()
            self.crossover()
            self.mutate()

        for c in self.population:
            evaluate_fitness(img_np, mask_np, c, self.seg_model, alpha=self.alpha, beta=self.beta)
        return max(self.population, key=lambda c: c.fitness if c.fitness is not None else -np.inf)
