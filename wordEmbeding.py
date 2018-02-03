#coding:utf8
__author__ = 'jmh081701'
import  tensorflow as tf
import  numpy as np
import  os
import  re
import  sys
import  bz2
import  random
import  collections
class Data_Provider(object):
    def __init__(self,cache_dir,data_dir,vocabulary_size=5000):
        '''
        :param cache_dir: 存储中间结果的缓存目录
        :param data_dir:  语料库目录
        :param vocabulary_size:  期望保留的字典大小
        :return:
        '''
        self._cache_dir=os.path.expanduser(cache_dir)
        self._sections_path =self._cache_dir+"//"+"sections.bz2"
        self._vocabulary_path=self._cache_dir+"//"+"vocab.bz2"
        #if os.path.isfile(self._sections_path)==False:
            #不存在sections文件,则重新建立
        self.read_data_section(data_dir)
        #if os.path.isfile(self._vocabulary_path)==False:
        self.build_vocabulary(vocabulary_size)
        with open(self._vocabulary_path,'rt',encoding='utf8') as vocabulary:
            self.vocabulary =vocabulary.read()
            self.vocabulary=self.vocabulary.strip().split()
            self.vocabulary2index={word:index for index,word in enumerate(self.vocabulary)}
            self.index2vocabulary={index:word for index,word in enumerate(self.vocabulary)}
    def encode(self,word):
        #返回某个词语在词汇表里面的下标
        if not word in self.vocabulary2index:
            word='<undefined>'
        return self.vocabulary2index[word]

    def decode(self,index):
        #返回index对应的词语
        return self.index2vocabulary[index]
    def vocabulary_size(self):
        #返回词汇表的大小
        return len(self.vocabulary)
    def build_vocabulary(self,vocabulary_size):
        #建立词汇表
        count=collections.Counter()
        with open(self._sections_path,"rt",encoding='utf8') as data:
            for eachline in data:
                words=eachline.strip().split(' ')
                count.update(words)
        most_common=count.most_common(vocabulary_size-1)
        common=[x[0] for x in most_common]
        common.append('<undefined>')
        #<underfined>是指那些出现次数特别少,或者特殊字符的词
        with open(self._vocabulary_path,'wt',encoding='utf8') as vocabulary:
            vocabulary.write(" ".join(common)+"\n")

    def read_data_section(self,data_dir):
        #从磁盘中读取源数据,遍历data_dir的每一个文件，每个文件是一篇新闻、短文.data_section应该是做过数据清洗的,里面只包含有效的word。
        #word与word之间用逗号隔开
        #
        for each in os.walk(data_dir):
            files=each[2]
            with open(self._sections_path,"wt",encoding='utf8') as sections:
                for file in files:
                    with open(data_dir+"//"+file,"rt",encoding='gbk') as section:
                        words=[]
                        for eachline in section:
                            words+=eachline.strip().split(" ")
                        sections.write(" ".join(words)+"\n") #把每篇文章拼接成一行

    def skip_gram(self,pages,max_context):
        #随机从当前位置的前(1,max_context)和后(1,max_context)组成训练对
        for page in pages:
            page=page.strip().split()
            for index,word in enumerate(page):
                context=random.randint(1,max_context)
                for pre in page[max(0,index-context):index]:
                    #前context个word
                    yield self.encode(word),self.encode(pre)
                for back in page[index+1:min(index+context,len(page)-1)]:
                    #后context个word
                    yield self.encode(word),self.encode(back)

    def next_batch(self,batch_size,max_context):
        with open(self._sections_path,encoding='utf8') as pages:
            gram=self.skip_gram(pages,max_context)
            while True:
                current=np.zeros(batch_size)
                target =np.zeros(batch_size)
                for i in range(batch_size):
                    current[i],target[i]=next(gram)
                yield current,target


class EmbedingModel(object):
    def __init__(self,current,target,vocabulary_size=1000,max_context=10,embedding_size=250,contrastive_example=100,learning_rate=0.01,momentum=0.5,batch_size=1000):
        self.x=current      #当前的词语
        self.y=target       #当前词语的上下文
        self.vocabulary_size=vocabulary_size
        self.max_context=max_context    #上下文大小
        self.embedding_size=embedding_size  #词向量的列长
        self.contrastive_example=contrastive_example  #噪声对比模型加速训练，而不是使用hierarchical-sigmoid
        self.momentum=momentum
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.embedding=tf.Variable(tf.random_uniform(shape=[vocabulary_size,embedding_size],minval=-0.5,maxval=0.5))
        embedded =tf.nn.embedding_lookup(self.embedding,self.x)
        weigh =tf.Variable(tf.truncated_normal(shape=[self.vocabulary_size,self.embedding_size]))
        bias =tf.Variable(tf.zeros(shape=[self.vocabulary_size]))
        #噪声对比模型加速训练，而不是使用hierarchical-sigmoid
        target=tf.expand_dims(self.y,1)
        self.cost=tf.reduce_mean(tf.nn.nce_loss(weigh,bias,target,embedded,self.contrastive_example,self.vocabulary_size))
        self.optimizer=tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.cost)


if __name__ == '__main__':
    dataset=Data_Provider(".//temp",".//passages",vocabulary_size=120)
    x=tf.placeholder(shape=[None],dtype=tf.int32,name='inputx')
    y=tf.placeholder(shape=[None],dtype=tf.float32,name='inputy')
    model =EmbedingModel(x,y,vocabulary_size=dataset.vocabulary_size(),batch_size=2000,embedding_size=4)
    sess =tf.Session()
    sess.run(tf.initialize_all_variables())
    batch_generator=dataset.next_batch(batch_size=model.batch_size,max_context=model.max_context)
    for current,target in batch_generator:
        sess.run(model.optimizer,feed_dict={x:current,y:target})
    embedding=sess.run(model.embedding)
    #词向量
    print(dataset.vocabulary)
    print(embedding,len(embedding))

