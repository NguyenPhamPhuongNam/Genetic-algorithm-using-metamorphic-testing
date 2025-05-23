# --- 5) Lớp Chromosome ---
class Chromosome:
    def __init__(self, img_shape, transformations=None):
        self.img_shape = img_shape
        if transformations is None:
            length = random.randint(2, 10)
            self.transformations = [generate_random_subvector(img_shape) for _ in range(length)]
        else:
            self.transformations = transformations
        self.fitness = None