from mod.utils.file_oper import *


def make_index(image_dir):
    ''' Make an index file for the dataset'''
    for folder in filter(lambda p: p.is_dir(), image_dir.iterdir()):
        txt = image_dir.parent / (folder.stem + '.txt')
        with open(txt, 'w') as f:
            for file in tqdm(list(folder.iterdir()), desc=txt.name):
                file = './' + str(file).replace('\\', '/')
                f.write(f'{file}\n')


def count_bbox(label_dir, n_cls):
    ''' Count the number of bounding boxes for each category'''
    data = {}
    for file in tqdm(list(label_dir.glob('**/*.txt')), desc='Count BBox'):
        with open(file) as f:
            bboxes = list(map(lambda row: int(row.split()[0]), f.readlines()))
            cnt = [0] * n_cls
            for i in bboxes: cnt[i] += 1
            data[file.name] = cnt
    return pd.DataFrame(data).T


def prune_dataset(image_dir, label_dir, cls_pool=None, min_n_boxes=1):
    ''' Pruning of the dataset in YOLOv7 format
        cls_pool: The indexes of the category to keep
        min_n_boxes: Minimum number of bounding boxes per image'''
    if cls_pool: cls_pool = {cls: i for i, cls in enumerate(
        range(cls_pool) if isinstance(cls_pool, int) else cls_pool)}
    # Read the image and clear any image that has no corresponding txt
    for image_folder in filter(lambda p: p.is_dir(), image_dir.iterdir()):
        unlink_count = 0
        label_folder = label_dir / image_folder.stem
        # Creating a progress bar
        pbar = tqdm(list(image_folder.iterdir()))
        for image in pbar:
            label = label_folder / (image.stem + '.txt')
            temp = label.with_suffix('.tmp')
            unlink_flag = False
            # Read tag files
            if label.is_file():
                with open(label) as f:
                    bboxes = f.readlines()
                # Filter out labels
                if cls_pool: bboxes = list(filter(
                    lambda bbox: int(bbox.split()[0]) in cls_pool.keys(), bboxes))
                if len(bboxes) >= min_n_boxes:
                    # Write temporary labels
                    if cls_pool:
                        with open(temp, 'w') as f:
                            for bbox in bboxes:
                                attr = bbox.split()
                                attr[0] = str(cls_pool[int(attr[0])])
                                f.write(' '.join(attr) + '\n')
                else:
                    unlink_flag = True
            else:
                # The tag file is empty
                unlink_flag = True
            if unlink_flag:
                # Delete the tag file, the image
                for file in (image, label): file.unlink(missing_ok=True)
                # Count the amount of deleted data
                unlink_count += 1
            prune_rate = unlink_count / len(pbar) * 100
            pbar.set_description(f'{image_folder.stem} Pruning Rate {prune_rate:.2f} %')
    # Make the temporary file overwrite the original file
    temp_files = list(label_dir.glob('**/*.tmp'))
    if temp_files:
        input('Type anything to start rewriting the label')
        for temp in tqdm(temp_files, desc='Overwrite Labels'):
            txt = temp.with_suffix('.txt')
            txt.unlink(missing_ok=True)
            temp.rename(txt)


def get_train_data(project, weighted=False, excel=True):
    ''' 解析 YOLOv7 result.txt'''
    with open(project / 'results.txt') as f:
        data = [list(map(eval, line.split()[2: -3]))
                for line in f.readlines()]
    data = list(map(lambda x: x[:3] + x[-4:], data))
    # 为数据附加列标签
    df = pd.DataFrame(
        data, columns=['box', 'obj', 'cls', 'Precision',
                       'Recall', 'AP-50', 'AP']
    )
    if not weighted:
        with open(project / 'hyp.yaml') as yl:
            hyp = yaml.load(yl.read(), Loader=yaml.Loader)
        # 使用权值将损失值还原
        for key in df.columns[:3]: df[key] /= hyp[key]
    # 将各个指标化成百分数
    for key in df.columns[3:]: df[key] *= 100
    if excel: excel_dump(df, project / f'{project.name}.xls')
    return df


def get_best_info(root=Path('runs'), weight=[0., 0., .1, .9]):
    ''' weight: P, R, AP-50, AP'''
    info = []
    for project in map(lambda txt: txt.parent,
                       root.glob('**/results.txt')):
        df = get_train_data(project)
        df = df[df.columns[3:]]
        fitness = (df * weight).sum(axis=1)
        # 输出最好轮次的结果
        res = df.iloc[fitness.argmax()]
        res.name = project.name
        res['Fitness'] = fitness.max()
        res['Size'] = get_size(project / 'weights/best.pt', unit='MB')
        info.append(res)
    return pd.DataFrame(info)


def get_best_curve(project, weight=[0., 0., .8, .2]):
    ''' 得到每个阶段的最优模型'''
    df = get_train_data(project, weighted=True, excel=False)
    df = df[df.columns[3:]]
    df['Fitness'] = (df * weight).sum(axis=1)
    for i in range(1, len(df)):
        if df['Fitness'][i] < df['Fitness'][i - 1]:
            df.iloc[i] = df.iloc[i - 1]
    return df


def clear_project(root=Path('runs'), blacklist=('pred', 'train', 'events')):
    ''' 清理训练项目中的冗余项'''
    for project in map(lambda txt: txt.parent,
                       root.glob('**/results.txt')):
        # 清理冗余图像
        for file in project.iterdir():
            if list(filter(lambda s: s in file.stem,
                           blacklist)): file.unlink()
        # 清理权重文件
        weights = project / 'weights'
        if weights.is_dir():
            temps = list(weights.glob('epoch*'))
            keep = ['best', 'last'] + ([max(temps).stem] if temps else [])
            for w in filter(lambda w: w.stem not in keep,
                            weights.iterdir()): w.unlink()


class Comparator:
    ''' 训练数据比较器'''
    _colors = np.array(['violet', 'deepskyblue', 'gold', 'darkorange'])
    _items = np.array(['AP-75', 'AP-50', 'AP'])

    def __init__(self, proposed: dict,
                 base: Path,
                 start_epoch: int = 0):
        self._start_epoch = start_epoch
        self._index = 0
        # 读取相关的训练数据
        for key in proposed:
            proposed[key] = get_train_data(proposed[key])[self._items][start_epoch:]
        self._proposed = proposed
        self._base = get_train_data(base)[self._items][start_epoch:]
        # 与 baseline 的比较
        self._diff = proposed.copy()
        for key in self._diff:
            self._diff[key] = (self._diff[key] - self._base).dropna()
        self.process()

    def _get_x(self, data):
        return np.arange(len(data)) + self._start_epoch

    def _zero_line(self, x):
        plt.plot(x, np.zeros_like(x), color='gray', linestyle='--', alpha=0.7)

    def _standard_coord(self, xlabel=True):
        self._index += 1
        fig = plt.subplot(*self.shape, self._index)
        for key in 'right', 'top':
            fig.spines[key].set_color('None')
        if xlabel: plt.xlabel('epoch')

    def _item_curve(self, pro_key):
        pro = self._proposed[pro_key]
        x = self._get_x(pro)
        base = self._base[:len(x)]
        # 绘制 AP-75, AP-50, AP 的曲线
        for i, item in enumerate(self._items):
            self._standard_coord()
            plt.plot(x, pro[item], color='deepskyblue', label=pro_key)
            # baseline
            plt.plot(x, base[item], color='darkorange', label='baseline')
            # 坐标轴设置
            plt.ylabel(item)
            plt.legend()

    def _diff_curve(self, pro_key, limit=None):
        diff = self._diff[pro_key]
        x = self._get_x(diff)
        self._standard_coord()
        # 绘制 AP-75, AP-50, AP 的差距曲线
        for item, color in zip(self._items, self._colors[[1, 0, 3]]):
            plt.plot(x, diff[item], color=color, label=item)
        self._zero_line(x)
        plt.ylabel('difference')
        plt.ylim(limit)
        plt.legend()

    def _ap_diff(self, limit=None):
        self._standard_coord()
        for i, key in enumerate(self._diff):
            diff = self._diff[key]['AP']
            x = self._get_x(diff)
            plt.plot(x, diff, color=self._colors[i], label=key)
        x = self._get_x(max(self._diff.values(), key=len))
        self._zero_line(x)
        plt.ylabel(f'{item} difference')
        plt.ylim(limit)
        plt.legend()
        # 绘制箱线图: 1
        x = np.arange(len(self._diff) + 1) + 0.5
        self._standard_coord(False)
        data = {key: self._diff[key]['AP'][:40] for key in self._diff}
        self._zero_line(x)
        boxplot(data.values(), labels=data.keys(), colors=self._colors)
        plt.ylim(limit)
        # 绘制箱线图: 2
        self._standard_coord(False)
        data = {key: self._diff[key]['AP'][40:] for key in self._diff}
        self._zero_line(x)
        boxplot(data.values(), labels=data.keys(), colors=self._colors)
        plt.ylim(limit)

    def process(self, diff_only=True):
        self.shape = [1 if diff_only else len(self._proposed), 3]
        # 设置完 shape 参数后, 开始绘图
        '''for key in self._proposed:
            if not diff_only: self._item_curve(key)
            self._diff_curve(key, [-1, 1])
            plt.title(key)'''
        self._ap_diff([-0.7, 0.3])
        plt.show()


def plot_one_box(x, img, color, label=None, line_thickness=None):
    ''' Drawing the bounding box'''
    tl = line_thickness or max([round(0.003 * min(img.shape[:2])), 1])  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def get_color(n_color):
    heat = np.linspace(0, 255, n_color).round().astype(np.uint8)[None]
    return cv2.applyColorMap(heat, cv2.COLORMAP_RAINBOW)[0].tolist()


def parse_label(image_dir, label_dir, detect_dir=None,
                show=False, category=None, color=None):
    ''' image_dir: Original image directory
        label_dir: Tag file directory
        detect_dir: Detect result directory
        category: List of category names
        color: Colors for each category'''
    if (detect_dir or show) and not color:
        color = get_color(len(category))
    if detect_dir and not detect_dir.is_dir(): detect_dir.mkdir()
    # Take the images one by one
    for img_file in tqdm(list(image_dir.iterdir())):
        txt = label_dir / img_file.with_suffix('.txt').name
        if txt.is_file():
            img = cv2.imread(str(img_file))
            h, w = img.shape[:2]
            # Resolve bounding boxes
            with open(txt) as f:
                for cls, *xywh in list(map(lambda s: list(map(eval, s.split())), f.readlines())):
                    xywh, conf = (xywh[:4], xywh[-1]) if len(xywh) == 5 else (xywh, None)
                    xywh = np.array(xywh)
                    # xywh -> xyxy
                    xywh[:2] -= xywh[2:] / 2
                    xywh[2:] += xywh[:2]
                    xywh[::2] *= w
                    xywh[1::2] *= h
                    xyxy = xywh
                    yield img_file.name, cls, xyxy, conf
                    # Draw the bounding box
                    if detect_dir or show:
                        plot_one_box(xyxy, img, color=color[cls],
                                     label=category[cls] + (f' {conf:.2f}' if conf else ''))
            # Store the image
            if detect_dir: cv2.imwrite(str(detect_dir / img_file.name), img)
            if show:
                cv2.imshow('show', img)
                cv2.waitKey(0)
