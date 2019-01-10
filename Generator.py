class StandardIterator(Iterator):
    def __init__(self, primary_dir, primary_files, labels=None,
                 batch_size=64, shuffle=True, seed=None,
                 secondary_dir=None, secondary_files=None):
        self.primary_dir = primary_dir
        self.file_names = primary_files + secondary_files if secondary_files else primary_files
        self.boundary = len(primary_files)
        self.secondary_dir = secondary_dir
        self.labels = labels
        super(StandardIterator, self).__init__(len(self.file_names), batch_size, shuffle, seed)

    def update(self, labels):
        with self.lock:
            self.labels = labels

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            batch_x = np.zeros((current_batch_size,4,IMAGE_SIZE,IMAGE_SIZE,3), dtype=K.floatx())
            batch_y = np.zeros((current_batch_size,37), dtype=K.floatx()) if self.labels else None

        for i, j in enumerate(index_array):
            is_primary = j < self.boundary
            fname = self.file_names[j]
            file_dir = self.primary_dir if is_primary else self.secondary_dir
            fpath = os.path.join(file_dir, fname)
            img = io.imread(fpath)
            img = img[70:-70,70:-70]
            img = resize(img, (IMAGE_SIZE,IMAGE_SIZE), mode='reflect')
            x = img_as_float(img)
            arrs = [np.rot90(x, a) for a in range(4)]
            x = np.stack(arrs,0)
            batch_x[i] = x
            if not (batch_y is None):
                batch_y[i] = np.reshape(self.labels[int(fname[:-4])], (-1,37))
        return (batch_x, batch_y) if (not (batch_y is None)) else batch_x

class MixedIterator(object):
    def __init__(self, train_dir, train_files, validation_files,
                 test_dir, test_files, train_labels, batch_split):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_files = train_files
        self.validation_files = validation_files
        self.test_files = test_files
        self.train_labels = train_labels
        self.pseudo_labels = {}
        self.train_gen = StandardIterator(train_dir, train_files, train_labels, batch_size=batch_split[0])
        self.pseudo_gen = StandardIterator(train_dir, validation_files, batch_size=batch_split[1],
                                           secondary_dir=self.test_dir, secondary_files=self.test_files)
        self.lock = threading.Lock()

    def reset(self):
        self.train_gen.reset()
        self.pseudo_gen.reset()

    def get_pseudo_labels(self):
        return self.pseudo_labels

    def update(self, model):
        with self.lock:
            pred_gen = StandardIterator(self.train_dir, self.validation_files, shuffle=False, batch_size=60)
            new_labels = model.predict_generator(pred_gen, len(self.validation_files)//60, verbose=1)
            for i, j in enumerate(self.validation_files):
                name = int(j[:-4])
                self.pseudo_labels[name] = new_labels[i,:]
            pred_gen = StandardIterator(self.test_dir, self.test_files, shuffle=False, batch_size=35)
            new_labels = model.predict_generator(pred_gen, len(self.test_files)//35, verbose=1)
            for i, j in enumerate(self.test_files):
                name = int(j[:-4])
                self.pseudo_labels[name] = new_labels[i,:]
            self.pseudo_gen.update(self.pseudo_labels)

    def __next__(self):
        return self.next()

    def next(self):
        with self.lock:
            a = next(self.train_gen)
            if len(self.pseudo_labels) > 0:
                b = next(self.pseudo_gen)
                return (np.concatenate((a[0],b[0])), np.concatenate((a[1],b[1])))
            else:
                return a
