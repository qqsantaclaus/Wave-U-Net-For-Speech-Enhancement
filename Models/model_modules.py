class AttentionPoolLayer:
    def __init__(self,
                 name,
                 channel_size=128,
                 attention_size=16,
                 num_heads=8,
                 global_cond_size=0,
                 sample_factor=2,
                 use_biases=False,
                 histograms=False,
                 training=True,
                 weightnorm=False,
                 residual=False,
                 learnable_up=False,
                 antialias=False,
                 attention_type="dot-product"):  
        self.name = name
        self.channel_size = channel_size
        self.attention_size = attention_size
        self.num_heads = num_heads
        self.residual = residual
        self.sample_factor = sample_factor
        
        self.use_biases = use_biases
        self.histograms = histograms
        self.global_cond_size = global_cond_size
        self.weightnorm = weightnorm
        self.training = training
        self.learnable_up = learnable_up
        self.attention_type = attention_type
        self.antialias = antialias
        
        if self.attention_type in ATTENTION_TYPES:
            att_head = ATTENTION_TYPES[self.attention_type]
        else:
            att_head = ATTENTION_TYPES["dot-product"]
            print("Attention type not recognized. Default to dot-product")
        self.att_head = att_head
        
        self.attention_map = []
        
        print("AttentionPoolLayer", {"residual": self.residual, 
                                     "sample_factor": self.sample_factor, 
                                     "learnable_up": self.learnable_up})
        
        self.variables = self._create_variables()
        
    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''
        if self.weightnorm:
            create_func = create_weightnorm_variable
        else:
            create_func = create_variable
            
        with tf.variable_scope(self.name):
            if self.sample_factor>1:
                if self.antialias:
                    print("Antialias")
                    self.project_down = AntialiasDownPoolLayer(pad_type='reflect', filt_size=self.sample_factor//2, channels=self.channel_size, pad_off=0)
                else:
                    self.project_down = DownPoolLayer()
            
            self.heads = []
            for i in range(self.num_heads):
                self.heads.append(
                    self.att_head("head_"+str(i),
                                     channel_size=self.channel_size,
                                     attention_size=self.attention_size,
                                     global_cond_size=self.global_cond_size,
                                     use_biases=self.use_biases,
                                     histograms=self.histograms,
                                     training=self.training,
                                     weightnorm=self.weightnorm)
                                )
            self.aggregation = ConvLayer("_agg",
                                         filter_width=1,
                                         channel_size=(self.attention_size * self.num_heads, self.channel_size),
                                         stride=1,
                                         dilation=1,
                                         global_cond_size=self.global_cond_size,
                                         use_biases=self.use_biases,
                                         histograms=self.histograms,
                                         training=self.training,
                                         weightnorm=self.weightnorm)
            if self.sample_factor>1:
                if self.learnable_up:
                    self.project_up = ConvTransposeLayer("_up",
                                                     filter_width=self.sample_factor,
                                                     channel_size=(self.channel_size, self.channel_size),
                                                     stride=self.sample_factor,
                                                     dilation=1,
                                                     global_cond_size=0,
                                                     use_biases=self.use_biases,
                                                     histograms=self.histograms,
                                                     training=self.training,
                                                     weightnorm=self.weightnorm)
                else:
                    self.project_up = UpLayer()
                                    
            if self.residual:
                self.scale_param = create_bias_variable('scale', [1,], value=0.0)
                        
    # batch x timesteps x channel
    def _create_layers(self, in_sample, lc_tensor=None, gc_tensor=None):
        outputs = []
        with tf.name_scope(self.name):
            if self.sample_factor>1:
                if self.antialias:
                    down_sample = self.project_down._create_layers(in_sample, pre_window=self.sample_factor, pre_stride=2, stride=self.sample_factor//2)
                else:
                    down_sample = self.project_down._create_layers(in_sample, self.sample_factor)
            else:
                down_sample = in_sample
            
            for i in range(self.num_heads):
                tmp_out = self.heads[i]._create_layers(down_sample, lc_tensor=lc_tensor, gc_tensor=gc_tensor)
                outputs.append(tmp_out)    
                self.attention_map.append(self.heads[i].attention_map)
                
            concat_outputs = tf.concat(outputs, axis=-1)
            
            if self.sample_factor>1 and self.learnable_up:
                activation = tf.nn.relu
            else:
                activation = None
            
            result = self.aggregation._create_layers(concat_outputs, gc_cond_tensor=gc_tensor, activation=activation)
            if self.sample_factor>1:
                if self.learnable_up:
                    result = self.project_up._create_layers(result, tf.shape(in_sample), 
                                                            gc_cond_tensor=None, activation=None)
                else:
                    result = self.project_up._create_layers(result, self.sample_factor, tf.shape(in_sample)[1])

            if self.residual:
                result = in_sample + self.scale_param * result
                
            tf.summary.scalar('scale_param', self.scale_param[0])
            
        return result