quantize_node_list = []


def onnx_runtime(model_path,image_files):
    '''
    Helper function run input image,and output each node tensor to calibration.
    parameter model_path: the onnx model
    parameter image_files: calibrate input images
    return: 
    '''
    sess = rt.InferenceSession(model_path)
    Input_name = sess.get_inputs()[0].name
    model_outputs = sess.get_outputs()
    print(len(model_outputs)) 
    # 1.对每一个需要量化的node 计算其输入的tensor中最大值
    for i,image in enumerate(image_files):
        img = cv2.imread(image)
        img = cv2.resize(img,(224,224))
        img = np.transpose(img,(2,0,1))
        img = img.astype('float32')/255
        img = img.reshape(1,224,224,3)
        start_time = datetime.datetime.now()
        for node in quantize_node_list:
            Output_name = node.output_name
            res = sess.run([Output_name],{Input_name:img})
            node.initial_input_max(np.array(res).flatten())
        end_time = datetime.datetime.now()
        print('it`s cost :', (end_time - start_time))
        if i % 100 == 0:
            print('loop stage 1 : %d/%d' % (i,len(image_files)))

    # calculate statistic node scope and interval distribution
    # 2.计算统计值分布的间隔用最大值除2048，即间隔的大小用于还原计算T
    for node in quantize_node_list:
        node.initial_input_distubution_interval()

    # for each nodes
    # collect histograms of activations
    # 3.得到每个node的数据分布，对于一个node得到的是在（0，max）划分2048块每个块内数据落在其中的数量统计值，假设区间[0,1]有20个数值落在里面。
    print('\n Collect histograms of activations: ')
    for i, image in enumerate(image_files):
        img = cv2.imread(image)
        img = cv2.resize(img,(224,224))
        img = np.transpose(img,(2,0,1))
        #print(img.shape)
        img = img.astype('float32')/255
        img = img.reshape(1,224,224,3)
        for node in quantize_node_list:
            Output_name = node.output_name
            res = sess.run([Output_name],{Input_name:img})
            node.initial_histograms(np.array(res).flatten())
        if i % 100 == 0:
            print('loop stage 2 : %d/%d' % (i,len(image_files)))

    # calculate threshold with KL divergence
    # 4. 核心计算KL散度
    for node in quantize_node_list:
        node.quantize_input()

    return None


#################################################
class Node:
    def __init__(self) -> None:
        pass

   # 1. 对每一个需要量化的node 计算其输入的tensor中最大值
    def initial_input_max(self, input_data):
        # get the max value of input
        max_val = np.max(input_data)
        min_val = np.min(input_data)
        self.input_max = max(self.input_max, max(abs(max_val), abs(min_val)))
   # 2.计算统计值分布的间隔用最大值除2048，即间隔的大小用于还原计算T
    def initial_input_distubution_interval(self):
        self.input_distubution_interval = STATISTIC * self.input_max / INTERVAL_NUM
        print("%-20s max_val : %-10.8f distribution_intervals : %-10.8f" % (self.node_name, self.input_max, self.input_distubution_interval))
   #3.得到每个node的数据分布，对于一个node得到的是在（0，max）划分2048块每个块内数据落在其中的数量统计值，假设区间[0,1]有20个数值落在里面。
    def initial_histograms(self, input_data):
        # collect histogram of every group channel input
        th = self.input_max
        # hist:Number of values in the interval for each hist,hist_edge:array of dtype float for interval. range: change the max and min value for inputdata. 
        hist, hist_edge = np.histogram(input_data, bins=INTERVAL_NUM, range=(0, th))
        self.input_distubution += hist
    
    def quantize_input(self):
        # calculate threshold  
        distribution = np.array(self.input_distubution)
        # pick threshold which minimizes KL divergence
        threshold_bin = threshold_distribution(distribution) 
        self.input_threshold = threshold_bin
        threshold = (threshold_bin + 0.5) * self.input_distubution_interval
        # get the activation calibration value
        self.input_scale = QUANTIZE_NUM / threshold

def threshold_distribution(distribution, target_bin=128):
    """
    Return the best threshold value. 
    Args:
        distribution: list, activations has been processed by histogram and normalize,size is 2048
        target_bin: int, the num of bin that is used by quantize, Int8 default value is 128
    Returns:
        target_threshold: int, num of bin with the minimum KL 
    """   
    distribution = distribution[1:]
    length = distribution.size
    threshold_sum = sum(distribution[target_bin:])
    kl_divergence = np.zeros(length - target_bin)
    # 遍历从128到2048开始搜索
    for threshold in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])

        # generate reference distribution p
        # 得到p比较简单，遍历的前threshold-1个组，将最后所有的组累加到threshold-1组上。
        p = sliced_nd_hist.copy()
        p[threshold-1] += threshold_sum
        threshold_sum = threshold_sum - distribution[threshold]

        # is_nonzeros[k] indicates whether hist[k] is nonzero
        # 判断p中元素是否有0存在，得到的is_nonzeros=[1,1,1,1,0,....]类似的array
        is_nonzeros = (p != 0).astype(np.int64)
        # 
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = sliced_nd_hist.size // target_bin
        
        # merge hist into num_quantized_bins bins
        # 这里是量化的原理，并不是数值的fp32-int8，只是将数据分布合并到128个组中，注意理解
        for j in range(target_bin):
            start = j * num_merged_bins #按照组的大小得到新组（128个组）前后位置
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()#属于同组的累加起来
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()#最后末尾的数据全部累加到新组的最后一组中
        
        # expand quantized_bins into p.size bins. compare with quantizated_bins merge, 
        that is inverse process
        # 将量化后的组重新扩大到与p相同大小的范围，就是按照前面量化的过程逆过来计算。
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins #找起始位置
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins #计算终止位置
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)# 把数据平均分配，这是逆过程差异的地方，只能平均分配。
        q[p == 0] = 0
        # p = _smooth_distribution(p) # with some bugs, need to fix
        # q = _smooth_distribution(q)
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001
        
        # calculate kl_divergence between q and p
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = min_kl_divergence + target_bin

    return threshold_value
