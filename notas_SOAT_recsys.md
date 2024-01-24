# Pesquisa SOTA RecSys

Buscaremos referências à trabalhos que apresentam sistemas de recomendação baseados em redes neuronais.
Queremos descobrir que métodos que utilizam Redes Neuronais para recomendação são considerados estado da arte em 2023.
Após esta etapa, estes serão avaliados de acordo com a framework utilizada na dissertação.

Os métodos que buscamos **não** devem levar em consideração informação adicional (variáveis além de interações usuário-item), ou serem incrementais. Mais, devem consumir feedback implícito.

---

Métodos de redes neuronais utilizados em ambos os ultimos artigos sobre problemas de reprodução e progresso em RecSys:
- MultiVAE
- NeuMF
Destes, o multivae é mais escalavel e tem melhor desempenho em ambos os estudos!

---

Os métodos que eram estado da arte anteriormente (e são razoavelmente reproduzíveis) são:

* Do paper *A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research* (https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation):
    * 2015: Collaborative Deep Learning for Recommender Systems (CDL): Hao Wang, Naiyan Wang, and Dit-Yan Yeung. Collaborative deep learning for recommender systems. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’15), pages 1235–1244, 2015.
    * 2017:
        * Collaborative Variational Autoencoder for Recommender Systems: Xiaopeng Li and James She. Collaborative variational autoencoder for recommender systems. In Proceedings of the 23th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’17), pages 305–314, 2017
        * Neural Collaborative Filtering: Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural collaborative filtering. In Proceedings of the 26th International Conference on World Wide Web (WWW ’17), pages 173–182, 2017.
        * Deep Matrix Factorization Models for Recommender Systems: Hong-Jian Xue, Xinyu Dai, Jianbing Zhang, Shujian Huang, and Jiajun Chen. Deep matrix factorization models for recommender systems. In Proceedings of the 26th International Joint Conference on Artificial Intelligence, (IJCAI ’18), pages 3203–3209, 2017.
    * 2018:
        * Variational Autoencoders for Collaborative Filtering: Dawen Liang, Rahul G Krishnan, Matthew D Hoffman, and Tony Jebara. Variational autoencoders for collaborative filtering. In Proceedings of the 27th International Conference on World Wide Web (WWW ’18), pages 689–698, 2018.
        * NeuRec: On Nonlinear Transformation for Personalized Ranking: Shuai Zhang, Lina Yao, Aixin Sun, Sen Wang, Guodong Long, and Manqing Dong. Neurec: On nonlinear transformation for personalized ranking. In Proceedings of the 27th International Joint Conference on Artificial Intelligence, (IJCAI ’18), pages 3669–3675, 2018.
        * CoupledCF: Learning Explicit and Implicit User-item Couplings in Recommendation for Deep Collaborative Filtering: Quangui Zhang, Longbing Cao, Chengzhang Zhu, Zhiqiang Li, and Jinguang Sun. Coupledcf: Learning explicit and implicit user-item couplings in recommendation for deep collaborative filtering. In Proceedings of the 27th International Joint Conference on Artificial Intelligence, (IJCAI ’18), pages 3662–3668, 7 2018
        * DELF: A Dual-Embedding based Deep Latent Factor Model for Recommendation: Weiyu Cheng, Yanyan Shen, Yanmin Zhu, and Linpeng Huang. Delf: A dual-embedding based deep latent factor model for recommendation. In Proceedings of the 27th International Joint Conference on Artificial Intelligence, (IJCAI ’18), pages 3329–3335, 2018.
        * Outer Product-based Neural Collaborative Filtering: Xiangnan He, Xiaoyu Du, Xiang Wang, Feng Tian, Jinhui Tang, Tat-Seng Chua, Outer Product-based Neural Collaborative Filtering, In Proceedings of IJCAI'18.
        * Leveraging Meta-path based Context for top-n Recommendation with a Neural Co-attention Model: Binbin Hu, Chuan Shi, Wayne Xin Zhao, and Philip S Yu. Leveraging meta-path based context for top-n recommendation with a neural co-attention model. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’18), pages 1531–1540, 2018.
        * Collaborative Memory Network for Recommendation Systems: Travis Ebesu, Bin Shen, and Yi Fang. Collaborative memory network for recommendation systems. In Proceedings of the 41st International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’18), pages 515–524, 2018.
        * Spectral Collaborative Filtering: Lei Zheng, Chun-Ta Lu, Fei Jiang, Jiawei Zhang, and Philip S. Yu. Spectral collaborative filtering. In Proceedings of the 12th ACM Conference on Recommender Systems (RecSys ’18), pages 311–319, 2018.
        
* Nvidia implementation (https://nvidia-merlin.github.io/models/main/models_overview.html#matrix-factorization):
    * 2019:
        * Deep Learning Recommendation Model for Personalization and Recommendation Systems (2019): Naumov, Maxim, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, et al. “Deep Learning Recommendation Model for Personalization and Recommendation Systems.” ArXiv:1906.00091 [Cs], May 31, 2019. 
    * 2020:
        * DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-Scale Learning to Rank Systems. (2020). Wang, Ruoxi, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed H. Chi. “DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-Scale Learning to Rank Systems.” ArXiv:2008.13535 [Cs, Stat], October 20, 2020.

* Disponíveis no DeepRec (https://github.com/cheungdaven/DeepRec/tree/master/models/item_ranking):
    * 2009:
        * BPRMF: Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 2009.
    * 2013:
        * dssm (Deep Semantic Similarity Model with BPR): Huang, Po-Sen, et al. "Learning deep structured semantic models for web search using clickthrough data." Proceedings of the 22nd ACM international conference on Conference on information & knowledge management. ACM, 2013.
    * 2016:
        * CDAE: Wu, Yao, et al. "Collaborative denoising auto-encoders for top-n recommender systems." Proceedings of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 2016.
    * 2017:
        * CML: Hsieh, Cheng-Kang, et al. "Collaborative metric learning." Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2017.
        * gmf: He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2017.
        * jrl: Zhang, Yongfeng, et al. "Joint representation learning for top-n recommendation with heterogeneous information sources." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. ACM, 2017.
        * lrml: Latent Relational Metric Learning (LRML) WWW 2018. Authors - Yi Tay, Luu Anh Tuan, Siu Cheung Hui
        * mlp: He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2017.
        * neumf: He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2017.
    * dmf: seems unavailable
    * neurec: seems unavailable
    * widedeep: seems unavailable
