Î%
ý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02unknown8¼!
l
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Variable
e
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes

:*
dtype0

!network_arch/inner_encoder/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*2
shared_name#!network_arch/inner_encoder/kernel

5network_arch/inner_encoder/kernel/Read/ReadVariableOpReadVariableOp!network_arch/inner_encoder/kernel*
_output_shapes
:	*
dtype0

!network_arch/inner_decoder/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*2
shared_name#!network_arch/inner_decoder/kernel

5network_arch/inner_decoder/kernel/Read/ReadVariableOpReadVariableOp!network_arch/inner_decoder/kernel*
_output_shapes
:	*
dtype0
²
)network_arch/conv_res_block/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)network_arch/conv_res_block/conv1d/kernel
«
=network_arch/conv_res_block/conv1d/kernel/Read/ReadVariableOpReadVariableOp)network_arch/conv_res_block/conv1d/kernel*"
_output_shapes
:*
dtype0
¦
'network_arch/conv_res_block/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'network_arch/conv_res_block/conv1d/bias

;network_arch/conv_res_block/conv1d/bias/Read/ReadVariableOpReadVariableOp'network_arch/conv_res_block/conv1d/bias*
_output_shapes
:*
dtype0
¶
+network_arch/conv_res_block/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+network_arch/conv_res_block/conv1d_1/kernel
¯
?network_arch/conv_res_block/conv1d_1/kernel/Read/ReadVariableOpReadVariableOp+network_arch/conv_res_block/conv1d_1/kernel*"
_output_shapes
:*
dtype0
ª
)network_arch/conv_res_block/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)network_arch/conv_res_block/conv1d_1/bias
£
=network_arch/conv_res_block/conv1d_1/bias/Read/ReadVariableOpReadVariableOp)network_arch/conv_res_block/conv1d_1/bias*
_output_shapes
:*
dtype0
¶
+network_arch/conv_res_block/conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+network_arch/conv_res_block/conv1d_2/kernel
¯
?network_arch/conv_res_block/conv1d_2/kernel/Read/ReadVariableOpReadVariableOp+network_arch/conv_res_block/conv1d_2/kernel*"
_output_shapes
: *
dtype0
ª
)network_arch/conv_res_block/conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)network_arch/conv_res_block/conv1d_2/bias
£
=network_arch/conv_res_block/conv1d_2/bias/Read/ReadVariableOpReadVariableOp)network_arch/conv_res_block/conv1d_2/bias*
_output_shapes
: *
dtype0
¶
+network_arch/conv_res_block/conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*<
shared_name-+network_arch/conv_res_block/conv1d_3/kernel
¯
?network_arch/conv_res_block/conv1d_3/kernel/Read/ReadVariableOpReadVariableOp+network_arch/conv_res_block/conv1d_3/kernel*"
_output_shapes
: @*
dtype0
ª
)network_arch/conv_res_block/conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)network_arch/conv_res_block/conv1d_3/bias
£
=network_arch/conv_res_block/conv1d_3/bias/Read/ReadVariableOpReadVariableOp)network_arch/conv_res_block/conv1d_3/bias*
_output_shapes
:@*
dtype0
®
(network_arch/conv_res_block/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(network_arch/conv_res_block/dense/kernel
§
<network_arch/conv_res_block/dense/kernel/Read/ReadVariableOpReadVariableOp(network_arch/conv_res_block/dense/kernel* 
_output_shapes
:
*
dtype0
¥
&network_arch/conv_res_block/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&network_arch/conv_res_block/dense/bias

:network_arch/conv_res_block/dense/bias/Read/ReadVariableOpReadVariableOp&network_arch/conv_res_block/dense/bias*
_output_shapes	
:*
dtype0
²
*network_arch/conv_res_block/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*network_arch/conv_res_block/dense_1/kernel
«
>network_arch/conv_res_block/dense_1/kernel/Read/ReadVariableOpReadVariableOp*network_arch/conv_res_block/dense_1/kernel* 
_output_shapes
:
*
dtype0
©
(network_arch/conv_res_block/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(network_arch/conv_res_block/dense_1/bias
¢
<network_arch/conv_res_block/dense_1/bias/Read/ReadVariableOpReadVariableOp(network_arch/conv_res_block/dense_1/bias*
_output_shapes	
:*
dtype0
º
-network_arch/conv_res_block_1/conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-network_arch/conv_res_block_1/conv1d_4/kernel
³
Anetwork_arch/conv_res_block_1/conv1d_4/kernel/Read/ReadVariableOpReadVariableOp-network_arch/conv_res_block_1/conv1d_4/kernel*"
_output_shapes
:*
dtype0
®
+network_arch/conv_res_block_1/conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+network_arch/conv_res_block_1/conv1d_4/bias
§
?network_arch/conv_res_block_1/conv1d_4/bias/Read/ReadVariableOpReadVariableOp+network_arch/conv_res_block_1/conv1d_4/bias*
_output_shapes
:*
dtype0
º
-network_arch/conv_res_block_1/conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-network_arch/conv_res_block_1/conv1d_5/kernel
³
Anetwork_arch/conv_res_block_1/conv1d_5/kernel/Read/ReadVariableOpReadVariableOp-network_arch/conv_res_block_1/conv1d_5/kernel*"
_output_shapes
:*
dtype0
®
+network_arch/conv_res_block_1/conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+network_arch/conv_res_block_1/conv1d_5/bias
§
?network_arch/conv_res_block_1/conv1d_5/bias/Read/ReadVariableOpReadVariableOp+network_arch/conv_res_block_1/conv1d_5/bias*
_output_shapes
:*
dtype0
º
-network_arch/conv_res_block_1/conv1d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-network_arch/conv_res_block_1/conv1d_6/kernel
³
Anetwork_arch/conv_res_block_1/conv1d_6/kernel/Read/ReadVariableOpReadVariableOp-network_arch/conv_res_block_1/conv1d_6/kernel*"
_output_shapes
: *
dtype0
®
+network_arch/conv_res_block_1/conv1d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+network_arch/conv_res_block_1/conv1d_6/bias
§
?network_arch/conv_res_block_1/conv1d_6/bias/Read/ReadVariableOpReadVariableOp+network_arch/conv_res_block_1/conv1d_6/bias*
_output_shapes
: *
dtype0
º
-network_arch/conv_res_block_1/conv1d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*>
shared_name/-network_arch/conv_res_block_1/conv1d_7/kernel
³
Anetwork_arch/conv_res_block_1/conv1d_7/kernel/Read/ReadVariableOpReadVariableOp-network_arch/conv_res_block_1/conv1d_7/kernel*"
_output_shapes
: @*
dtype0
®
+network_arch/conv_res_block_1/conv1d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+network_arch/conv_res_block_1/conv1d_7/bias
§
?network_arch/conv_res_block_1/conv1d_7/bias/Read/ReadVariableOpReadVariableOp+network_arch/conv_res_block_1/conv1d_7/bias*
_output_shapes
:@*
dtype0
¶
,network_arch/conv_res_block_1/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,network_arch/conv_res_block_1/dense_2/kernel
¯
@network_arch/conv_res_block_1/dense_2/kernel/Read/ReadVariableOpReadVariableOp,network_arch/conv_res_block_1/dense_2/kernel* 
_output_shapes
:
*
dtype0
­
*network_arch/conv_res_block_1/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*network_arch/conv_res_block_1/dense_2/bias
¦
>network_arch/conv_res_block_1/dense_2/bias/Read/ReadVariableOpReadVariableOp*network_arch/conv_res_block_1/dense_2/bias*
_output_shapes	
:*
dtype0
¶
,network_arch/conv_res_block_1/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,network_arch/conv_res_block_1/dense_3/kernel
¯
@network_arch/conv_res_block_1/dense_3/kernel/Read/ReadVariableOpReadVariableOp,network_arch/conv_res_block_1/dense_3/kernel* 
_output_shapes
:
*
dtype0
­
*network_arch/conv_res_block_1/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*network_arch/conv_res_block_1/dense_3/bias
¦
>network_arch/conv_res_block_1/dense_3/bias/Read/ReadVariableOpReadVariableOp*network_arch/conv_res_block_1/dense_3/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ò\
valueÈ\BÅ\ B¾\
æ
outer_encoder
outer_decoder
inner_encoder
L
inner_decoder
inner_loss_weights
	optimizer
loss
	
signatures

regularization_losses
	variables
trainable_variables
	keras_api
u
conv_layers
dense_layers
	variables
regularization_losses
trainable_variables
	keras_api
u
conv_layers
dense_layers
	variables
regularization_losses
trainable_variables
	keras_api
^

kernel
	variables
regularization_losses
trainable_variables
	keras_api
:8
VARIABLE_VALUEVariableL/.ATTRIBUTES/VARIABLE_VALUE
^

kernel
 	variables
!regularization_losses
"trainable_variables
#	keras_api
 
 
 
 
 
Î
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719
820
921
:22
;23
24
25
26
Î
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719
820
921
:22
;23
24
25
26
­
<layer_regularization_losses
=non_trainable_variables
>metrics

regularization_losses
	variables
?layer_metrics

@layers
trainable_variables
1
A0
B1
C2
D3
E4
F5
G6

H0
I1
J2
V
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
 
V
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
­
Klayer_regularization_losses
Lnon_trainable_variables
Mmetrics
	variables
regularization_losses
Nlayer_metrics

Olayers
trainable_variables
1
P0
Q1
R2
S3
T4
U5
V6

W0
X1
Y2
V
00
11
22
33
44
55
66
77
88
99
:10
;11
 
V
00
11
22
33
44
55
66
77
88
99
:10
;11
­
Zlayer_regularization_losses
[non_trainable_variables
\metrics
	variables
regularization_losses
]layer_metrics

^layers
trainable_variables
fd
VARIABLE_VALUE!network_arch/inner_encoder/kernel/inner_encoder/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
­
_layer_regularization_losses
`non_trainable_variables
ametrics
	variables
regularization_losses
blayer_metrics

clayers
trainable_variables
fd
VARIABLE_VALUE!network_arch/inner_decoder/kernel/inner_decoder/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
­
dlayer_regularization_losses
enon_trainable_variables
fmetrics
 	variables
!regularization_losses
glayer_metrics

hlayers
"trainable_variables
ec
VARIABLE_VALUE)network_arch/conv_res_block/conv1d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'network_arch/conv_res_block/conv1d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+network_arch/conv_res_block/conv1d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)network_arch/conv_res_block/conv1d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+network_arch/conv_res_block/conv1d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)network_arch/conv_res_block/conv1d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+network_arch/conv_res_block/conv1d_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)network_arch/conv_res_block/conv1d_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE(network_arch/conv_res_block/dense/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&network_arch/conv_res_block/dense/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*network_arch/conv_res_block/dense_1/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(network_arch/conv_res_block/dense_1/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network_arch/conv_res_block_1/conv1d_4/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+network_arch/conv_res_block_1/conv1d_4/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network_arch/conv_res_block_1/conv1d_5/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+network_arch/conv_res_block_1/conv1d_5/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network_arch/conv_res_block_1/conv1d_6/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+network_arch/conv_res_block_1/conv1d_6/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network_arch/conv_res_block_1/conv1d_7/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+network_arch/conv_res_block_1/conv1d_7/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,network_arch/conv_res_block_1/dense_2/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*network_arch/conv_res_block_1/dense_2/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,network_arch/conv_res_block_1/dense_3/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*network_arch/conv_res_block_1/dense_3/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 

0
1
2
3
h

$kernel
%bias
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
R
m	variables
nregularization_losses
otrainable_variables
p	keras_api
h

&kernel
'bias
q	variables
rregularization_losses
strainable_variables
t	keras_api
R
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
h

(kernel
)bias
y	variables
zregularization_losses
{trainable_variables
|	keras_api
S
}	variables
~regularization_losses
trainable_variables
	keras_api
l

*kernel
+bias
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
l

,kernel
-bias
	variables
regularization_losses
trainable_variables
	keras_api
l

.kernel
/bias
	variables
regularization_losses
trainable_variables
	keras_api
 
 
 
 
F
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
l

0kernel
1bias
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
l

2kernel
3bias
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
 	keras_api
l

4kernel
5bias
¡	variables
¢regularization_losses
£trainable_variables
¤	keras_api
V
¥	variables
¦regularization_losses
§trainable_variables
¨	keras_api
l

6kernel
7bias
©	variables
ªregularization_losses
«trainable_variables
¬	keras_api
V
­	variables
®regularization_losses
¯trainable_variables
°	keras_api
l

8kernel
9bias
±	variables
²regularization_losses
³trainable_variables
´	keras_api
l

:kernel
;bias
µ	variables
¶regularization_losses
·trainable_variables
¸	keras_api
 
 
 
 
F
P0
Q1
R2
S3
T4
U5
V6
W7
X8
Y9
 
 
 
 
 
 
 
 
 
 

$0
%1
 

$0
%1
²
 ¹layer_regularization_losses
ºnon_trainable_variables
»metrics
i	variables
jregularization_losses
¼layer_metrics
½layers
ktrainable_variables
 
 
 
²
 ¾layer_regularization_losses
¿non_trainable_variables
Àmetrics
m	variables
nregularization_losses
Álayer_metrics
Âlayers
otrainable_variables

&0
'1
 

&0
'1
²
 Ãlayer_regularization_losses
Änon_trainable_variables
Åmetrics
q	variables
rregularization_losses
Ælayer_metrics
Çlayers
strainable_variables
 
 
 
²
 Èlayer_regularization_losses
Énon_trainable_variables
Êmetrics
u	variables
vregularization_losses
Ëlayer_metrics
Ìlayers
wtrainable_variables

(0
)1
 

(0
)1
²
 Ílayer_regularization_losses
Înon_trainable_variables
Ïmetrics
y	variables
zregularization_losses
Ðlayer_metrics
Ñlayers
{trainable_variables
 
 
 
²
 Òlayer_regularization_losses
Ónon_trainable_variables
Ômetrics
}	variables
~regularization_losses
Õlayer_metrics
Ölayers
trainable_variables

*0
+1
 

*0
+1
µ
 ×layer_regularization_losses
Ønon_trainable_variables
Ùmetrics
	variables
regularization_losses
Úlayer_metrics
Ûlayers
trainable_variables
 
 
 
µ
 Ülayer_regularization_losses
Ýnon_trainable_variables
Þmetrics
	variables
regularization_losses
ßlayer_metrics
àlayers
trainable_variables

,0
-1
 

,0
-1
µ
 álayer_regularization_losses
ânon_trainable_variables
ãmetrics
	variables
regularization_losses
älayer_metrics
ålayers
trainable_variables

.0
/1
 

.0
/1
µ
 ælayer_regularization_losses
çnon_trainable_variables
èmetrics
	variables
regularization_losses
élayer_metrics
êlayers
trainable_variables

00
11
 

00
11
µ
 ëlayer_regularization_losses
ìnon_trainable_variables
ímetrics
	variables
regularization_losses
îlayer_metrics
ïlayers
trainable_variables
 
 
 
µ
 ðlayer_regularization_losses
ñnon_trainable_variables
òmetrics
	variables
regularization_losses
ólayer_metrics
ôlayers
trainable_variables

20
31
 

20
31
µ
 õlayer_regularization_losses
önon_trainable_variables
÷metrics
	variables
regularization_losses
ølayer_metrics
ùlayers
trainable_variables
 
 
 
µ
 úlayer_regularization_losses
ûnon_trainable_variables
ümetrics
	variables
regularization_losses
ýlayer_metrics
þlayers
trainable_variables

40
51
 

40
51
µ
 ÿlayer_regularization_losses
non_trainable_variables
metrics
¡	variables
¢regularization_losses
layer_metrics
layers
£trainable_variables
 
 
 
µ
 layer_regularization_losses
non_trainable_variables
metrics
¥	variables
¦regularization_losses
layer_metrics
layers
§trainable_variables

60
71
 

60
71
µ
 layer_regularization_losses
non_trainable_variables
metrics
©	variables
ªregularization_losses
layer_metrics
layers
«trainable_variables
 
 
 
µ
 layer_regularization_losses
non_trainable_variables
metrics
­	variables
®regularization_losses
layer_metrics
layers
¯trainable_variables

80
91
 

80
91
µ
 layer_regularization_losses
non_trainable_variables
metrics
±	variables
²regularization_losses
layer_metrics
layers
³trainable_variables

:0
;1
 

:0
;1
µ
 layer_regularization_losses
non_trainable_variables
metrics
µ	variables
¶regularization_losses
layer_metrics
layers
·trainable_variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_1Placeholder*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*
dtype0*!
shape:ÿÿÿÿÿÿÿÿÿ3
÷
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1)network_arch/conv_res_block/conv1d/kernel'network_arch/conv_res_block/conv1d/bias+network_arch/conv_res_block/conv1d_1/kernel)network_arch/conv_res_block/conv1d_1/bias+network_arch/conv_res_block/conv1d_2/kernel)network_arch/conv_res_block/conv1d_2/bias+network_arch/conv_res_block/conv1d_3/kernel)network_arch/conv_res_block/conv1d_3/bias(network_arch/conv_res_block/dense/kernel&network_arch/conv_res_block/dense/bias*network_arch/conv_res_block/dense_1/kernel(network_arch/conv_res_block/dense_1/bias!network_arch/inner_encoder/kernel!network_arch/inner_decoder/kernel-network_arch/conv_res_block_1/conv1d_4/kernel+network_arch/conv_res_block_1/conv1d_4/bias-network_arch/conv_res_block_1/conv1d_5/kernel+network_arch/conv_res_block_1/conv1d_5/bias-network_arch/conv_res_block_1/conv1d_6/kernel+network_arch/conv_res_block_1/conv1d_6/bias-network_arch/conv_res_block_1/conv1d_7/kernel+network_arch/conv_res_block_1/conv1d_7/bias,network_arch/conv_res_block_1/dense_2/kernel*network_arch/conv_res_block_1/dense_2/bias,network_arch/conv_res_block_1/dense_3/kernel*network_arch/conv_res_block_1/dense_3/biasVariable*'
Tin 
2*
Tout
2*\
_output_shapesJ
H:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ2*=
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*/
f*R(
&__inference_signature_wrapper_13477229
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ª
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp5network_arch/inner_encoder/kernel/Read/ReadVariableOp5network_arch/inner_decoder/kernel/Read/ReadVariableOp=network_arch/conv_res_block/conv1d/kernel/Read/ReadVariableOp;network_arch/conv_res_block/conv1d/bias/Read/ReadVariableOp?network_arch/conv_res_block/conv1d_1/kernel/Read/ReadVariableOp=network_arch/conv_res_block/conv1d_1/bias/Read/ReadVariableOp?network_arch/conv_res_block/conv1d_2/kernel/Read/ReadVariableOp=network_arch/conv_res_block/conv1d_2/bias/Read/ReadVariableOp?network_arch/conv_res_block/conv1d_3/kernel/Read/ReadVariableOp=network_arch/conv_res_block/conv1d_3/bias/Read/ReadVariableOp<network_arch/conv_res_block/dense/kernel/Read/ReadVariableOp:network_arch/conv_res_block/dense/bias/Read/ReadVariableOp>network_arch/conv_res_block/dense_1/kernel/Read/ReadVariableOp<network_arch/conv_res_block/dense_1/bias/Read/ReadVariableOpAnetwork_arch/conv_res_block_1/conv1d_4/kernel/Read/ReadVariableOp?network_arch/conv_res_block_1/conv1d_4/bias/Read/ReadVariableOpAnetwork_arch/conv_res_block_1/conv1d_5/kernel/Read/ReadVariableOp?network_arch/conv_res_block_1/conv1d_5/bias/Read/ReadVariableOpAnetwork_arch/conv_res_block_1/conv1d_6/kernel/Read/ReadVariableOp?network_arch/conv_res_block_1/conv1d_6/bias/Read/ReadVariableOpAnetwork_arch/conv_res_block_1/conv1d_7/kernel/Read/ReadVariableOp?network_arch/conv_res_block_1/conv1d_7/bias/Read/ReadVariableOp@network_arch/conv_res_block_1/dense_2/kernel/Read/ReadVariableOp>network_arch/conv_res_block_1/dense_2/bias/Read/ReadVariableOp@network_arch/conv_res_block_1/dense_3/kernel/Read/ReadVariableOp>network_arch/conv_res_block_1/dense_3/bias/Read/ReadVariableOpConst*(
Tin!
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8**
f%R#
!__inference__traced_save_13478059

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable!network_arch/inner_encoder/kernel!network_arch/inner_decoder/kernel)network_arch/conv_res_block/conv1d/kernel'network_arch/conv_res_block/conv1d/bias+network_arch/conv_res_block/conv1d_1/kernel)network_arch/conv_res_block/conv1d_1/bias+network_arch/conv_res_block/conv1d_2/kernel)network_arch/conv_res_block/conv1d_2/bias+network_arch/conv_res_block/conv1d_3/kernel)network_arch/conv_res_block/conv1d_3/bias(network_arch/conv_res_block/dense/kernel&network_arch/conv_res_block/dense/bias*network_arch/conv_res_block/dense_1/kernel(network_arch/conv_res_block/dense_1/bias-network_arch/conv_res_block_1/conv1d_4/kernel+network_arch/conv_res_block_1/conv1d_4/bias-network_arch/conv_res_block_1/conv1d_5/kernel+network_arch/conv_res_block_1/conv1d_5/bias-network_arch/conv_res_block_1/conv1d_6/kernel+network_arch/conv_res_block_1/conv1d_6/bias-network_arch/conv_res_block_1/conv1d_7/kernel+network_arch/conv_res_block_1/conv1d_7/bias,network_arch/conv_res_block_1/dense_2/kernel*network_arch/conv_res_block_1/dense_2/bias,network_arch/conv_res_block_1/dense_3/kernel*network_arch/conv_res_block_1/dense_3/bias*'
Tin 
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*-
f(R&
$__inference__traced_restore_13478152î
¯

+__inference_conv1d_6_layer_call_fn_13477549

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv1d_6_layer_call_and_return_conditional_losses_134775392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
è
h
__inference_loss_fn_3_134777235
1kernel_regularizer_square_readvariableop_resource
identityÊ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/add]
IdentityIdentitykernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
È
v
0__inference_inner_decoder_layer_call_fn_13469383

inputs
unknown
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
÷
m
Q__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_13477508

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsº
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸Ò
¿
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_13470137

inputs8
4conv1d_4_conv1d_expanddims_1_readvariableop_resource,
(conv1d_4_biasadd_readvariableop_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource,
(conv1d_5_biasadd_readvariableop_resource8
4conv1d_6_conv1d_expanddims_1_readvariableop_resource,
(conv1d_6_biasadd_readvariableop_resource8
4conv1d_7_conv1d_expanddims_1_readvariableop_resource,
(conv1d_7_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim~

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDims
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_4/conv1d/ExpandDims/dim¿
conv1d_4/conv1d/ExpandDims
ExpandDimsExpandDims:output:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_4/conv1d/ExpandDimsÓ
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dimÛ
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_4/conv1d/ExpandDims_1Û
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d_4/conv1d¥
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d_4/conv1d/Squeeze§
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_4/BiasAdd/ReadVariableOp±
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_4/BiasAddx
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_4/Relu
conv1d_4/IdentityIdentityconv1d_4/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_4/Identity
"average_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_3/ExpandDims/dimÒ
average_pooling1d_3/ExpandDims
ExpandDimsconv1d_4/Identity:output:0+average_pooling1d_3/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
average_pooling1d_3/ExpandDimsä
average_pooling1d_3/AvgPoolAvgPool'average_pooling1d_3/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
average_pooling1d_3/AvgPool¸
average_pooling1d_3/SqueezeSqueeze$average_pooling1d_3/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
average_pooling1d_3/Squeeze¤
average_pooling1d_3/IdentityIdentity$average_pooling1d_3/Squeeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
average_pooling1d_3/Identity
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_5/conv1d/ExpandDims/dimÐ
conv1d_5/conv1d/ExpandDims
ExpandDims%average_pooling1d_3/Identity:output:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_5/conv1d/ExpandDimsÓ
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dimÛ
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_5/conv1d/ExpandDims_1Ú
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv1d_5/conv1d¤
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
conv1d_5/conv1d/Squeeze§
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_5/BiasAdd/ReadVariableOp°
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_5/BiasAddw
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_5/Relu
conv1d_5/IdentityIdentityconv1d_5/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_5/Identity
"average_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_4/ExpandDims/dimÑ
average_pooling1d_4/ExpandDims
ExpandDimsconv1d_5/Identity:output:0+average_pooling1d_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
average_pooling1d_4/ExpandDimsä
average_pooling1d_4/AvgPoolAvgPool'average_pooling1d_4/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_4/AvgPool¸
average_pooling1d_4/SqueezeSqueeze$average_pooling1d_4/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2
average_pooling1d_4/Squeeze¤
average_pooling1d_4/IdentityIdentity$average_pooling1d_4/Squeeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
average_pooling1d_4/Identity
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_6/conv1d/ExpandDims/dimÐ
conv1d_6/conv1d/ExpandDims
ExpandDims%average_pooling1d_4/Identity:output:0'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_6/conv1d/ExpandDimsÓ
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_6/conv1d/ExpandDims_1/dimÛ
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_6/conv1d/ExpandDims_1Ú
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv1d_6/conv1d¤
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
squeeze_dims
2
conv1d_6/conv1d/Squeeze§
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_6/BiasAdd/ReadVariableOp°
conv1d_6/BiasAddBiasAdd conv1d_6/conv1d/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_6/BiasAddw
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_6/Relu
conv1d_6/IdentityIdentityconv1d_6/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_6/Identity
"average_pooling1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_5/ExpandDims/dimÑ
average_pooling1d_5/ExpandDims
ExpandDimsconv1d_6/Identity:output:0+average_pooling1d_5/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2 
average_pooling1d_5/ExpandDimsä
average_pooling1d_5/AvgPoolAvgPool'average_pooling1d_5/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_5/AvgPool¸
average_pooling1d_5/SqueezeSqueeze$average_pooling1d_5/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2
average_pooling1d_5/Squeeze¤
average_pooling1d_5/IdentityIdentity$average_pooling1d_5/Squeeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
average_pooling1d_5/Identity
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_7/conv1d/ExpandDims/dimÐ
conv1d_7/conv1d/ExpandDims
ExpandDims%average_pooling1d_5/Identity:output:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_7/conv1d/ExpandDimsÓ
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_7/conv1d/ExpandDims_1/dimÛ
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_7/conv1d/ExpandDims_1Ú
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv1d_7/conv1d¤
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
conv1d_7/conv1d/Squeeze§
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_7/BiasAdd/ReadVariableOp°
conv1d_7/BiasAddBiasAdd conv1d_7/conv1d/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_7/BiasAddw
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_7/Relu
conv1d_7/IdentityIdentityconv1d_7/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_7/Identitys
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_1/Const
flatten_1/ReshapeReshapeconv1d_7/Identity:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_1/Reshape
flatten_1/IdentityIdentityflatten_1/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_1/Identity§
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_2/MatMul/ReadVariableOp¡
dense_2/MatMulMatMulflatten_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¢
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu
dense_2/IdentityIdentitydense_2/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Identity§
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMuldense_2/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul¥
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¢
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAdd}
dense_3/IdentityIdentitydense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/Identityi
addAddV2inputsdense_3/Identity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
Onetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02Q
Onetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square/ReadVariableOp
@network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/SquareSquareWnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2B
@network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square×
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Const¶
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/SumSumDnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square:y:0Hnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/SumÇ
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22A
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mul/x¸
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mulMulHnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mul/x:output:0Fnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mulÇ
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/add/xµ
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/addAddV2Hnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/add/x:output:0Anetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/add
Onetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02Q
Onetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square/ReadVariableOp
@network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/SquareSquareWnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2B
@network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square×
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Const¶
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/SumSumDnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square:y:0Hnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/SumÇ
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22A
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mul/x¸
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mulMulHnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mul/x:output:0Fnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mulÇ
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/add/xµ
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/addAddV2Hnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/add/x:output:0Anetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/add
Onetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02Q
Onetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square/ReadVariableOp
@network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/SquareSquareWnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2B
@network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square×
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Const¶
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/SumSumDnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square:y:0Hnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/SumÇ
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22A
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mul/x¸
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mulMulHnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mul/x:output:0Fnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mulÇ
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/add/xµ
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/addAddV2Hnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/add/x:output:0Anetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/add
Onetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02Q
Onetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square/ReadVariableOp
@network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/SquareSquareWnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2B
@network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square×
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Const¶
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/SumSumDnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square:y:0Hnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/SumÇ
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22A
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mul/x¸
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mulMulHnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mul/x:output:0Fnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mulÇ
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/add/xµ
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/addAddV2Hnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/add/x:output:0Anetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/add
Nnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02P
Nnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Square/ReadVariableOp
?network_arch/conv_res_block_1/dense_2/kernel/Regularizer/SquareSquareVnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2A
?network_arch/conv_res_block_1/dense_2/kernel/Regularizer/SquareÑ
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2@
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/Const²
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/SumSumCnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Square:y:0Gnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/SumÅ
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22@
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mul/x´
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mulMulGnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/mul/x:output:0Enetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mulÅ
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/add/x±
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/addAddV2Gnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/add/x:output:0@network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/add
Nnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02P
Nnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Square/ReadVariableOp
?network_arch/conv_res_block_1/dense_3/kernel/Regularizer/SquareSquareVnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2A
?network_arch/conv_res_block_1/dense_3/kernel/Regularizer/SquareÑ
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2@
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/Const²
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/SumSumCnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Square:y:0Gnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/SumÅ
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22@
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mul/x´
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mulMulGnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/mul/x:output:0Enetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mulÅ
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/add/x±
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/addAddV2Gnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/add/x:output:0@network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/add\
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ:::::::::::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ù

1__inference_conv_res_block_layer_call_fn_13469183

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*U
fPRN
L__inference_conv_res_block_layer_call_and_return_conditional_losses_134691662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
è
h
__inference_loss_fn_8_134777885
1kernel_regularizer_square_readvariableop_resource
identityÊ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/add]
IdentityIdentitykernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
¯

+__inference_conv1d_1_layer_call_fn_13477314

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_134773042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
¯

+__inference_conv1d_4_layer_call_fn_13477449

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv1d_4_layer_call_and_return_conditional_losses_134774392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

¨
+__inference_restored_function_body_13476991
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25
identity

identity_1

identity_2¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*'
Tin 
2*
Tout
2*\
_output_shapesJ
H:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ2*=
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_network_arch_layer_call_and_return_conditional_losses_134749902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3:::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ç
¿
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_13468847

inputs8
4conv1d_4_conv1d_expanddims_1_readvariableop_resource,
(conv1d_4_biasadd_readvariableop_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource,
(conv1d_5_biasadd_readvariableop_resource8
4conv1d_6_conv1d_expanddims_1_readvariableop_resource,
(conv1d_6_biasadd_readvariableop_resource8
4conv1d_7_conv1d_expanddims_1_readvariableop_resource,
(conv1d_7_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim~

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDims
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_4/conv1d/ExpandDims/dim¿
conv1d_4/conv1d/ExpandDims
ExpandDimsExpandDims:output:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_4/conv1d/ExpandDimsÓ
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dimÛ
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_4/conv1d/ExpandDims_1Û
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d_4/conv1d¥
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d_4/conv1d/Squeeze§
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_4/BiasAdd/ReadVariableOp±
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_4/BiasAddx
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_4/Relu
"average_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_3/ExpandDims/dimÓ
average_pooling1d_3/ExpandDims
ExpandDimsconv1d_4/Relu:activations:0+average_pooling1d_3/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
average_pooling1d_3/ExpandDimsä
average_pooling1d_3/AvgPoolAvgPool'average_pooling1d_3/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
average_pooling1d_3/AvgPool¸
average_pooling1d_3/SqueezeSqueeze$average_pooling1d_3/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
average_pooling1d_3/Squeeze
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_5/conv1d/ExpandDims/dimÏ
conv1d_5/conv1d/ExpandDims
ExpandDims$average_pooling1d_3/Squeeze:output:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_5/conv1d/ExpandDimsÓ
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dimÛ
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_5/conv1d/ExpandDims_1Ú
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv1d_5/conv1d¤
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
conv1d_5/conv1d/Squeeze§
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_5/BiasAdd/ReadVariableOp°
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_5/BiasAddw
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_5/Relu
"average_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_4/ExpandDims/dimÒ
average_pooling1d_4/ExpandDims
ExpandDimsconv1d_5/Relu:activations:0+average_pooling1d_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
average_pooling1d_4/ExpandDimsä
average_pooling1d_4/AvgPoolAvgPool'average_pooling1d_4/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_4/AvgPool¸
average_pooling1d_4/SqueezeSqueeze$average_pooling1d_4/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2
average_pooling1d_4/Squeeze
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_6/conv1d/ExpandDims/dimÏ
conv1d_6/conv1d/ExpandDims
ExpandDims$average_pooling1d_4/Squeeze:output:0'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_6/conv1d/ExpandDimsÓ
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_6/conv1d/ExpandDims_1/dimÛ
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_6/conv1d/ExpandDims_1Ú
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv1d_6/conv1d¤
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
squeeze_dims
2
conv1d_6/conv1d/Squeeze§
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_6/BiasAdd/ReadVariableOp°
conv1d_6/BiasAddBiasAdd conv1d_6/conv1d/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_6/BiasAddw
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_6/Relu
"average_pooling1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_5/ExpandDims/dimÒ
average_pooling1d_5/ExpandDims
ExpandDimsconv1d_6/Relu:activations:0+average_pooling1d_5/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2 
average_pooling1d_5/ExpandDimsä
average_pooling1d_5/AvgPoolAvgPool'average_pooling1d_5/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_5/AvgPool¸
average_pooling1d_5/SqueezeSqueeze$average_pooling1d_5/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2
average_pooling1d_5/Squeeze
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_7/conv1d/ExpandDims/dimÏ
conv1d_7/conv1d/ExpandDims
ExpandDims$average_pooling1d_5/Squeeze:output:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_7/conv1d/ExpandDimsÓ
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_7/conv1d/ExpandDims_1/dimÛ
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_7/conv1d/ExpandDims_1Ú
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv1d_7/conv1d¤
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
conv1d_7/conv1d/Squeeze§
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_7/BiasAdd/ReadVariableOp°
conv1d_7/BiasAddBiasAdd conv1d_7/conv1d/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_7/BiasAddw
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_7/Relus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_1/Const
flatten_1/ReshapeReshapeconv1d_7/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_1/Reshape§
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_2/MatMul/ReadVariableOp 
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¢
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu§
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_3/MatMul/ReadVariableOp 
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul¥
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¢
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAddh
addAddV2inputsdense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
Onetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02Q
Onetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square/ReadVariableOp
@network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/SquareSquareWnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2B
@network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square×
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Const¶
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/SumSumDnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square:y:0Hnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/SumÇ
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22A
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mul/x¸
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mulMulHnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mul/x:output:0Fnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mulÇ
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/add/xµ
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/addAddV2Hnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/add/x:output:0Anetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/add
Onetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02Q
Onetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square/ReadVariableOp
@network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/SquareSquareWnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2B
@network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square×
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Const¶
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/SumSumDnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square:y:0Hnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/SumÇ
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22A
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mul/x¸
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mulMulHnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mul/x:output:0Fnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mulÇ
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/add/xµ
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/addAddV2Hnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/add/x:output:0Anetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/add
Onetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02Q
Onetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square/ReadVariableOp
@network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/SquareSquareWnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2B
@network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square×
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Const¶
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/SumSumDnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square:y:0Hnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/SumÇ
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22A
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mul/x¸
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mulMulHnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mul/x:output:0Fnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mulÇ
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/add/xµ
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/addAddV2Hnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/add/x:output:0Anetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/add
Onetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02Q
Onetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square/ReadVariableOp
@network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/SquareSquareWnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2B
@network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square×
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Const¶
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/SumSumDnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square:y:0Hnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/SumÇ
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22A
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mul/x¸
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mulMulHnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mul/x:output:0Fnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mulÇ
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/add/xµ
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/addAddV2Hnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/add/x:output:0Anetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/add
Nnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02P
Nnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Square/ReadVariableOp
?network_arch/conv_res_block_1/dense_2/kernel/Regularizer/SquareSquareVnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2A
?network_arch/conv_res_block_1/dense_2/kernel/Regularizer/SquareÑ
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2@
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/Const²
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/SumSumCnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Square:y:0Gnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/SumÅ
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22@
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mul/x´
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mulMulGnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/mul/x:output:0Enetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mulÅ
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/add/x±
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/addAddV2Gnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/add/x:output:0@network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/add
Nnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02P
Nnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Square/ReadVariableOp
?network_arch/conv_res_block_1/dense_3/kernel/Regularizer/SquareSquareVnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2A
?network_arch/conv_res_block_1/dense_3/kernel/Regularizer/SquareÑ
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2@
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/Const²
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/SumSumCnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Square:y:0Gnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/SumÅ
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22@
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mul/x´
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mulMulGnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/mul/x:output:0Enetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mulÅ
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/add/x±
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/addAddV2Gnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/add/x:output:0@network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/add\
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ:::::::::::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¯

+__inference_conv1d_3_layer_call_fn_13477414

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv1d_3_layer_call_and_return_conditional_losses_134774042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ã
R
6__inference_average_pooling1d_4_layer_call_fn_13477514

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Z
fURS
Q__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_134775082
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
h
__inference_loss_fn_9_134778015
1kernel_regularizer_square_readvariableop_resource
identityÊ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/add]
IdentityIdentitykernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ã|

$__inference__traced_restore_13478152
file_prefix
assignvariableop_variable8
4assignvariableop_1_network_arch_inner_encoder_kernel8
4assignvariableop_2_network_arch_inner_decoder_kernel@
<assignvariableop_3_network_arch_conv_res_block_conv1d_kernel>
:assignvariableop_4_network_arch_conv_res_block_conv1d_biasB
>assignvariableop_5_network_arch_conv_res_block_conv1d_1_kernel@
<assignvariableop_6_network_arch_conv_res_block_conv1d_1_biasB
>assignvariableop_7_network_arch_conv_res_block_conv1d_2_kernel@
<assignvariableop_8_network_arch_conv_res_block_conv1d_2_biasB
>assignvariableop_9_network_arch_conv_res_block_conv1d_3_kernelA
=assignvariableop_10_network_arch_conv_res_block_conv1d_3_bias@
<assignvariableop_11_network_arch_conv_res_block_dense_kernel>
:assignvariableop_12_network_arch_conv_res_block_dense_biasB
>assignvariableop_13_network_arch_conv_res_block_dense_1_kernel@
<assignvariableop_14_network_arch_conv_res_block_dense_1_biasE
Aassignvariableop_15_network_arch_conv_res_block_1_conv1d_4_kernelC
?assignvariableop_16_network_arch_conv_res_block_1_conv1d_4_biasE
Aassignvariableop_17_network_arch_conv_res_block_1_conv1d_5_kernelC
?assignvariableop_18_network_arch_conv_res_block_1_conv1d_5_biasE
Aassignvariableop_19_network_arch_conv_res_block_1_conv1d_6_kernelC
?assignvariableop_20_network_arch_conv_res_block_1_conv1d_6_biasE
Aassignvariableop_21_network_arch_conv_res_block_1_conv1d_7_kernelC
?assignvariableop_22_network_arch_conv_res_block_1_conv1d_7_biasD
@assignvariableop_23_network_arch_conv_res_block_1_dense_2_kernelB
>assignvariableop_24_network_arch_conv_res_block_1_dense_2_biasD
@assignvariableop_25_network_arch_conv_res_block_1_dense_3_kernelB
>assignvariableop_26_network_arch_conv_res_block_1_dense_3_bias
identity_28¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1×	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ã
valueÙBÖBL/.ATTRIBUTES/VARIABLE_VALUEB/inner_encoder/kernel/.ATTRIBUTES/VARIABLE_VALUEB/inner_decoder/kernel/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesÄ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices³
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1ª
AssignVariableOp_1AssignVariableOp4assignvariableop_1_network_arch_inner_encoder_kernelIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2ª
AssignVariableOp_2AssignVariableOp4assignvariableop_2_network_arch_inner_decoder_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3²
AssignVariableOp_3AssignVariableOp<assignvariableop_3_network_arch_conv_res_block_conv1d_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4°
AssignVariableOp_4AssignVariableOp:assignvariableop_4_network_arch_conv_res_block_conv1d_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5´
AssignVariableOp_5AssignVariableOp>assignvariableop_5_network_arch_conv_res_block_conv1d_1_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6²
AssignVariableOp_6AssignVariableOp<assignvariableop_6_network_arch_conv_res_block_conv1d_1_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7´
AssignVariableOp_7AssignVariableOp>assignvariableop_7_network_arch_conv_res_block_conv1d_2_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8²
AssignVariableOp_8AssignVariableOp<assignvariableop_8_network_arch_conv_res_block_conv1d_2_biasIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9´
AssignVariableOp_9AssignVariableOp>assignvariableop_9_network_arch_conv_res_block_conv1d_3_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10¶
AssignVariableOp_10AssignVariableOp=assignvariableop_10_network_arch_conv_res_block_conv1d_3_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11µ
AssignVariableOp_11AssignVariableOp<assignvariableop_11_network_arch_conv_res_block_dense_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12³
AssignVariableOp_12AssignVariableOp:assignvariableop_12_network_arch_conv_res_block_dense_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13·
AssignVariableOp_13AssignVariableOp>assignvariableop_13_network_arch_conv_res_block_dense_1_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14µ
AssignVariableOp_14AssignVariableOp<assignvariableop_14_network_arch_conv_res_block_dense_1_biasIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15º
AssignVariableOp_15AssignVariableOpAassignvariableop_15_network_arch_conv_res_block_1_conv1d_4_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16¸
AssignVariableOp_16AssignVariableOp?assignvariableop_16_network_arch_conv_res_block_1_conv1d_4_biasIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17º
AssignVariableOp_17AssignVariableOpAassignvariableop_17_network_arch_conv_res_block_1_conv1d_5_kernelIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18¸
AssignVariableOp_18AssignVariableOp?assignvariableop_18_network_arch_conv_res_block_1_conv1d_5_biasIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19º
AssignVariableOp_19AssignVariableOpAassignvariableop_19_network_arch_conv_res_block_1_conv1d_6_kernelIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20¸
AssignVariableOp_20AssignVariableOp?assignvariableop_20_network_arch_conv_res_block_1_conv1d_6_biasIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21º
AssignVariableOp_21AssignVariableOpAassignvariableop_21_network_arch_conv_res_block_1_conv1d_7_kernelIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22¸
AssignVariableOp_22AssignVariableOp?assignvariableop_22_network_arch_conv_res_block_1_conv1d_7_biasIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23¹
AssignVariableOp_23AssignVariableOp@assignvariableop_23_network_arch_conv_res_block_1_dense_2_kernelIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp>assignvariableop_24_network_arch_conv_res_block_1_dense_2_biasIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25¹
AssignVariableOp_25AssignVariableOp@assignvariableop_25_network_arch_conv_res_block_1_dense_3_kernelIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp>assignvariableop_26_network_arch_conv_res_block_1_dense_3_biasIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp°
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27½
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*
_input_shapesp
n: :::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Î
¹
D__inference_conv1d_layer_call_and_return_conditional_losses_13477254

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¿
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
ReluÄ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/adds
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ô

K__inference_inner_decoder_layer_call_and_return_conditional_losses_13469377

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulê
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpí
4network_arch/inner_decoder/kernel/Regularizer/SquareSquareKnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	26
4network_arch/inner_decoder/kernel/Regularizer/Square»
3network_arch/inner_decoder/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       25
3network_arch/inner_decoder/kernel/Regularizer/Const
1network_arch/inner_decoder/kernel/Regularizer/SumSum8network_arch/inner_decoder/kernel/Regularizer/Square:y:0<network_arch/inner_decoder/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/Sum¯
3network_arch/inner_decoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+225
3network_arch/inner_decoder/kernel/Regularizer/mul/x
1network_arch/inner_decoder/kernel/Regularizer/mulMul<network_arch/inner_decoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_decoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/mul¯
3network_arch/inner_decoder/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3network_arch/inner_decoder/kernel/Regularizer/add/x
1network_arch/inner_decoder/kernel/Regularizer/addAddV2<network_arch/inner_decoder/kernel/Regularizer/add/x:output:05network_arch/inner_decoder/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/adde
IdentityIdentityMatMul:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
ã
R
6__inference_average_pooling1d_3_layer_call_fn_13477464

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Z
fURS
Q__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_134774582
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
i
__inference_loss_fn_10_134778145
1kernel_regularizer_square_readvariableop_resource
identityÊ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*"
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/add]
IdentityIdentitykernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ð
»
F__inference_conv1d_3_layer_call_and_return_conditional_losses_13477404

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1¿
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
ReluÄ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/adds
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ô

K__inference_inner_encoder_layer_call_and_return_conditional_losses_13469014

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulê
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpí
4network_arch/inner_encoder/kernel/Regularizer/SquareSquareKnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	26
4network_arch/inner_encoder/kernel/Regularizer/Square»
3network_arch/inner_encoder/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       25
3network_arch/inner_encoder/kernel/Regularizer/Const
1network_arch/inner_encoder/kernel/Regularizer/SumSum8network_arch/inner_encoder/kernel/Regularizer/Square:y:0<network_arch/inner_encoder/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/Sum¯
3network_arch/inner_encoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+225
3network_arch/inner_encoder/kernel/Regularizer/mul/x
1network_arch/inner_encoder/kernel/Regularizer/mulMul<network_arch/inner_encoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_encoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/mul¯
3network_arch/inner_encoder/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3network_arch/inner_encoder/kernel/Regularizer/add/x
1network_arch/inner_encoder/kernel/Regularizer/addAddV2<network_arch/inner_encoder/kernel/Regularizer/add/x:output:05network_arch/inner_encoder/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/addd
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
÷
m
Q__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_13477373

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsº
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

K__inference_inner_decoder_layer_call_and_return_conditional_losses_13469411

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulê
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpí
4network_arch/inner_decoder/kernel/Regularizer/SquareSquareKnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	26
4network_arch/inner_decoder/kernel/Regularizer/Square»
3network_arch/inner_decoder/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       25
3network_arch/inner_decoder/kernel/Regularizer/Const
1network_arch/inner_decoder/kernel/Regularizer/SumSum8network_arch/inner_decoder/kernel/Regularizer/Square:y:0<network_arch/inner_decoder/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/Sum¯
3network_arch/inner_decoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+225
3network_arch/inner_decoder/kernel/Regularizer/mul/x
1network_arch/inner_decoder/kernel/Regularizer/mulMul<network_arch/inner_decoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_decoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/mul¯
3network_arch/inner_decoder/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3network_arch/inner_decoder/kernel/Regularizer/add/x
1network_arch/inner_decoder/kernel/Regularizer/addAddV2<network_arch/inner_decoder/kernel/Regularizer/add/x:output:05network_arch/inner_decoder/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/adde
IdentityIdentityMatMul:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
ß
P
4__inference_average_pooling1d_layer_call_fn_13477279

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_average_pooling1d_layer_call_and_return_conditional_losses_134772732
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
h
__inference_loss_fn_2_134777105
1kernel_regularizer_square_readvariableop_resource
identityÊ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/add]
IdentityIdentitykernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 

¬
/__inference_network_arch_layer_call_fn_13475134
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25
identity

identity_1

identity_2¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*'
Tin 
2*
Tout
2*\
_output_shapesJ
H:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ2*=
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_network_arch_layer_call_and_return_conditional_losses_134749902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3:::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¯

+__inference_conv1d_5_layer_call_fn_13477499

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv1d_5_layer_call_and_return_conditional_losses_134774892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
÷
m
Q__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_13477323

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsº
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
h
__inference_loss_fn_5_134777495
1kernel_regularizer_square_readvariableop_resource
identityÊ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*"
_output_shapes
: @*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/add]
IdentityIdentitykernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
è
h
__inference_loss_fn_4_134777365
1kernel_regularizer_square_readvariableop_resource
identityÊ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*"
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/add]
IdentityIdentitykernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Á
Å-
J__inference_network_arch_layer_call_and_return_conditional_losses_13474990
input_1
conv_res_block_3112520
conv_res_block_3112522
conv_res_block_3112524
conv_res_block_3112526
conv_res_block_3112528
conv_res_block_3112530
conv_res_block_3112532
conv_res_block_3112534
conv_res_block_3112536
conv_res_block_3112538
conv_res_block_3112540
conv_res_block_3112542
inner_encoder_3112570
inner_decoder_3112598
conv_res_block_1_3112773
conv_res_block_1_3112775
conv_res_block_1_3112777
conv_res_block_1_3112779
conv_res_block_1_3112781
conv_res_block_1_3112783
conv_res_block_1_3112785
conv_res_block_1_3112787
conv_res_block_1_3112789
conv_res_block_1_3112791
conv_res_block_1_3112793
conv_res_block_1_3112795"
matmul_readvariableop_resource
identity_100
identity_101
identity_102¢&conv_res_block/StatefulPartitionedCall¢(conv_res_block_1/StatefulPartitionedCall¢*conv_res_block_1_1/StatefulPartitionedCall¢+conv_res_block_1_10/StatefulPartitionedCall¢+conv_res_block_1_11/StatefulPartitionedCall¢+conv_res_block_1_12/StatefulPartitionedCall¢+conv_res_block_1_13/StatefulPartitionedCall¢+conv_res_block_1_14/StatefulPartitionedCall¢+conv_res_block_1_15/StatefulPartitionedCall¢+conv_res_block_1_16/StatefulPartitionedCall¢+conv_res_block_1_17/StatefulPartitionedCall¢+conv_res_block_1_18/StatefulPartitionedCall¢+conv_res_block_1_19/StatefulPartitionedCall¢*conv_res_block_1_2/StatefulPartitionedCall¢+conv_res_block_1_20/StatefulPartitionedCall¢+conv_res_block_1_21/StatefulPartitionedCall¢+conv_res_block_1_22/StatefulPartitionedCall¢+conv_res_block_1_23/StatefulPartitionedCall¢+conv_res_block_1_24/StatefulPartitionedCall¢+conv_res_block_1_25/StatefulPartitionedCall¢+conv_res_block_1_26/StatefulPartitionedCall¢+conv_res_block_1_27/StatefulPartitionedCall¢+conv_res_block_1_28/StatefulPartitionedCall¢+conv_res_block_1_29/StatefulPartitionedCall¢*conv_res_block_1_3/StatefulPartitionedCall¢+conv_res_block_1_30/StatefulPartitionedCall¢+conv_res_block_1_31/StatefulPartitionedCall¢+conv_res_block_1_32/StatefulPartitionedCall¢+conv_res_block_1_33/StatefulPartitionedCall¢+conv_res_block_1_34/StatefulPartitionedCall¢+conv_res_block_1_35/StatefulPartitionedCall¢+conv_res_block_1_36/StatefulPartitionedCall¢+conv_res_block_1_37/StatefulPartitionedCall¢+conv_res_block_1_38/StatefulPartitionedCall¢+conv_res_block_1_39/StatefulPartitionedCall¢*conv_res_block_1_4/StatefulPartitionedCall¢+conv_res_block_1_40/StatefulPartitionedCall¢+conv_res_block_1_41/StatefulPartitionedCall¢+conv_res_block_1_42/StatefulPartitionedCall¢+conv_res_block_1_43/StatefulPartitionedCall¢+conv_res_block_1_44/StatefulPartitionedCall¢+conv_res_block_1_45/StatefulPartitionedCall¢+conv_res_block_1_46/StatefulPartitionedCall¢+conv_res_block_1_47/StatefulPartitionedCall¢+conv_res_block_1_48/StatefulPartitionedCall¢+conv_res_block_1_49/StatefulPartitionedCall¢*conv_res_block_1_5/StatefulPartitionedCall¢+conv_res_block_1_50/StatefulPartitionedCall¢+conv_res_block_1_51/StatefulPartitionedCall¢*conv_res_block_1_6/StatefulPartitionedCall¢*conv_res_block_1_7/StatefulPartitionedCall¢*conv_res_block_1_8/StatefulPartitionedCall¢*conv_res_block_1_9/StatefulPartitionedCall¢(conv_res_block_2/StatefulPartitionedCall¢(conv_res_block_3/StatefulPartitionedCall¢(conv_res_block_4/StatefulPartitionedCall¢%inner_decoder/StatefulPartitionedCall¢'inner_decoder_1/StatefulPartitionedCall¢(inner_decoder_10/StatefulPartitionedCall¢(inner_decoder_11/StatefulPartitionedCall¢(inner_decoder_12/StatefulPartitionedCall¢(inner_decoder_13/StatefulPartitionedCall¢(inner_decoder_14/StatefulPartitionedCall¢(inner_decoder_15/StatefulPartitionedCall¢(inner_decoder_16/StatefulPartitionedCall¢(inner_decoder_17/StatefulPartitionedCall¢(inner_decoder_18/StatefulPartitionedCall¢(inner_decoder_19/StatefulPartitionedCall¢'inner_decoder_2/StatefulPartitionedCall¢(inner_decoder_20/StatefulPartitionedCall¢(inner_decoder_21/StatefulPartitionedCall¢(inner_decoder_22/StatefulPartitionedCall¢(inner_decoder_23/StatefulPartitionedCall¢(inner_decoder_24/StatefulPartitionedCall¢(inner_decoder_25/StatefulPartitionedCall¢(inner_decoder_26/StatefulPartitionedCall¢(inner_decoder_27/StatefulPartitionedCall¢(inner_decoder_28/StatefulPartitionedCall¢(inner_decoder_29/StatefulPartitionedCall¢'inner_decoder_3/StatefulPartitionedCall¢(inner_decoder_30/StatefulPartitionedCall¢(inner_decoder_31/StatefulPartitionedCall¢(inner_decoder_32/StatefulPartitionedCall¢(inner_decoder_33/StatefulPartitionedCall¢(inner_decoder_34/StatefulPartitionedCall¢(inner_decoder_35/StatefulPartitionedCall¢(inner_decoder_36/StatefulPartitionedCall¢(inner_decoder_37/StatefulPartitionedCall¢(inner_decoder_38/StatefulPartitionedCall¢(inner_decoder_39/StatefulPartitionedCall¢'inner_decoder_4/StatefulPartitionedCall¢(inner_decoder_40/StatefulPartitionedCall¢(inner_decoder_41/StatefulPartitionedCall¢(inner_decoder_42/StatefulPartitionedCall¢(inner_decoder_43/StatefulPartitionedCall¢(inner_decoder_44/StatefulPartitionedCall¢(inner_decoder_45/StatefulPartitionedCall¢(inner_decoder_46/StatefulPartitionedCall¢(inner_decoder_47/StatefulPartitionedCall¢(inner_decoder_48/StatefulPartitionedCall¢(inner_decoder_49/StatefulPartitionedCall¢'inner_decoder_5/StatefulPartitionedCall¢(inner_decoder_50/StatefulPartitionedCall¢'inner_decoder_6/StatefulPartitionedCall¢'inner_decoder_7/StatefulPartitionedCall¢'inner_decoder_8/StatefulPartitionedCall¢'inner_decoder_9/StatefulPartitionedCall¢%inner_encoder/StatefulPartitionedCall¢'inner_encoder_1/StatefulPartitionedCall¢'inner_encoder_2/StatefulPartitionedCall¢'inner_encoder_3/StatefulPartitionedCall
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2û
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_2/stack_2
strided_slice_2StridedSliceinput_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_3/stack_2
strided_slice_3StridedSliceinput_1strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_4/stack
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_4/stack_1
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_4/stack_2
strided_slice_4StridedSliceinput_1strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_4
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_5/stack
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_5/stack_1
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_5/stack_2
strided_slice_5StridedSliceinput_1strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_5
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_6/stack
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_6/stack_1
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_6/stack_2
strided_slice_6StridedSliceinput_1strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_6
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_7/stack
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_7/stack_1
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_7/stack_2
strided_slice_7StridedSliceinput_1strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_7
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_8/stack
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_8/stack_1
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_8/stack_2
strided_slice_8StridedSliceinput_1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_8
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_9/stack
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       2
strided_slice_9/stack_1
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_9/stack_2
strided_slice_9StridedSliceinput_1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_9
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       2
strided_slice_10/stack
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       2
strided_slice_10/stack_1
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_10/stack_2
strided_slice_10StridedSliceinput_1strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_10
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       2
strided_slice_11/stack
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_11/stack_1
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_11/stack_2
strided_slice_11StridedSliceinput_1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_11
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_12/stack
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_12/stack_1
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_12/stack_2
strided_slice_12StridedSliceinput_1strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_12
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_13/stack
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_13/stack_1
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_13/stack_2
strided_slice_13StridedSliceinput_1strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_13
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_14/stack
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_14/stack_1
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_14/stack_2
strided_slice_14StridedSliceinput_1strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_14
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_15/stack
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_15/stack_1
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_15/stack_2
strided_slice_15StridedSliceinput_1strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_15
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_16/stack
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_16/stack_1
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_16/stack_2
strided_slice_16StridedSliceinput_1strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_16
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_17/stack
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_17/stack_1
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_17/stack_2
strided_slice_17StridedSliceinput_1strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_17
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_18/stack
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_18/stack_1
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_18/stack_2
strided_slice_18StridedSliceinput_1strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_18
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_19/stack
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_19/stack_1
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_19/stack_2
strided_slice_19StridedSliceinput_1strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_19
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_20/stack
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_20/stack_1
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_20/stack_2
strided_slice_20StridedSliceinput_1strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_20
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_21/stack
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_21/stack_1
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_21/stack_2
strided_slice_21StridedSliceinput_1strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_21
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_22/stack
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_22/stack_1
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_22/stack_2
strided_slice_22StridedSliceinput_1strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_22
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_23/stack
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_23/stack_1
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_23/stack_2
strided_slice_23StridedSliceinput_1strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_23
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_24/stack
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_24/stack_1
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_24/stack_2
strided_slice_24StridedSliceinput_1strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_24
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_25/stack
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_25/stack_1
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_25/stack_2
strided_slice_25StridedSliceinput_1strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_25
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_26/stack
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_26/stack_1
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_26/stack_2
strided_slice_26StridedSliceinput_1strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_26
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_27/stack
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_27/stack_1
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_27/stack_2
strided_slice_27StridedSliceinput_1strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_27
strided_slice_28/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_28/stack
strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_28/stack_1
strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_28/stack_2
strided_slice_28StridedSliceinput_1strided_slice_28/stack:output:0!strided_slice_28/stack_1:output:0!strided_slice_28/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_28
strided_slice_29/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_29/stack
strided_slice_29/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_29/stack_1
strided_slice_29/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_29/stack_2
strided_slice_29StridedSliceinput_1strided_slice_29/stack:output:0!strided_slice_29/stack_1:output:0!strided_slice_29/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_29
strided_slice_30/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_30/stack
strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_30/stack_1
strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_30/stack_2
strided_slice_30StridedSliceinput_1strided_slice_30/stack:output:0!strided_slice_30/stack_1:output:0!strided_slice_30/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_30
strided_slice_31/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_31/stack
strided_slice_31/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_31/stack_1
strided_slice_31/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_31/stack_2
strided_slice_31StridedSliceinput_1strided_slice_31/stack:output:0!strided_slice_31/stack_1:output:0!strided_slice_31/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_31
strided_slice_32/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_32/stack
strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_32/stack_1
strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_32/stack_2
strided_slice_32StridedSliceinput_1strided_slice_32/stack:output:0!strided_slice_32/stack_1:output:0!strided_slice_32/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_32
strided_slice_33/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_33/stack
strided_slice_33/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    !       2
strided_slice_33/stack_1
strided_slice_33/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_33/stack_2
strided_slice_33StridedSliceinput_1strided_slice_33/stack:output:0!strided_slice_33/stack_1:output:0!strided_slice_33/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_33
strided_slice_34/stackConst*
_output_shapes
:*
dtype0*!
valueB"    !       2
strided_slice_34/stack
strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    "       2
strided_slice_34/stack_1
strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_34/stack_2
strided_slice_34StridedSliceinput_1strided_slice_34/stack:output:0!strided_slice_34/stack_1:output:0!strided_slice_34/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_34
strided_slice_35/stackConst*
_output_shapes
:*
dtype0*!
valueB"    "       2
strided_slice_35/stack
strided_slice_35/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    #       2
strided_slice_35/stack_1
strided_slice_35/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_35/stack_2
strided_slice_35StridedSliceinput_1strided_slice_35/stack:output:0!strided_slice_35/stack_1:output:0!strided_slice_35/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_35
strided_slice_36/stackConst*
_output_shapes
:*
dtype0*!
valueB"    #       2
strided_slice_36/stack
strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    $       2
strided_slice_36/stack_1
strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_36/stack_2
strided_slice_36StridedSliceinput_1strided_slice_36/stack:output:0!strided_slice_36/stack_1:output:0!strided_slice_36/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_36
strided_slice_37/stackConst*
_output_shapes
:*
dtype0*!
valueB"    $       2
strided_slice_37/stack
strided_slice_37/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    %       2
strided_slice_37/stack_1
strided_slice_37/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_37/stack_2
strided_slice_37StridedSliceinput_1strided_slice_37/stack:output:0!strided_slice_37/stack_1:output:0!strided_slice_37/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_37
strided_slice_38/stackConst*
_output_shapes
:*
dtype0*!
valueB"    %       2
strided_slice_38/stack
strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    &       2
strided_slice_38/stack_1
strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_38/stack_2
strided_slice_38StridedSliceinput_1strided_slice_38/stack:output:0!strided_slice_38/stack_1:output:0!strided_slice_38/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_38
strided_slice_39/stackConst*
_output_shapes
:*
dtype0*!
valueB"    &       2
strided_slice_39/stack
strided_slice_39/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    '       2
strided_slice_39/stack_1
strided_slice_39/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_39/stack_2
strided_slice_39StridedSliceinput_1strided_slice_39/stack:output:0!strided_slice_39/stack_1:output:0!strided_slice_39/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_39
strided_slice_40/stackConst*
_output_shapes
:*
dtype0*!
valueB"    '       2
strided_slice_40/stack
strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    (       2
strided_slice_40/stack_1
strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_40/stack_2
strided_slice_40StridedSliceinput_1strided_slice_40/stack:output:0!strided_slice_40/stack_1:output:0!strided_slice_40/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_40
strided_slice_41/stackConst*
_output_shapes
:*
dtype0*!
valueB"    (       2
strided_slice_41/stack
strided_slice_41/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    )       2
strided_slice_41/stack_1
strided_slice_41/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_41/stack_2
strided_slice_41StridedSliceinput_1strided_slice_41/stack:output:0!strided_slice_41/stack_1:output:0!strided_slice_41/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_41
strided_slice_42/stackConst*
_output_shapes
:*
dtype0*!
valueB"    )       2
strided_slice_42/stack
strided_slice_42/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    *       2
strided_slice_42/stack_1
strided_slice_42/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_42/stack_2
strided_slice_42StridedSliceinput_1strided_slice_42/stack:output:0!strided_slice_42/stack_1:output:0!strided_slice_42/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_42
strided_slice_43/stackConst*
_output_shapes
:*
dtype0*!
valueB"    *       2
strided_slice_43/stack
strided_slice_43/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    +       2
strided_slice_43/stack_1
strided_slice_43/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_43/stack_2
strided_slice_43StridedSliceinput_1strided_slice_43/stack:output:0!strided_slice_43/stack_1:output:0!strided_slice_43/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_43
strided_slice_44/stackConst*
_output_shapes
:*
dtype0*!
valueB"    +       2
strided_slice_44/stack
strided_slice_44/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    ,       2
strided_slice_44/stack_1
strided_slice_44/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_44/stack_2
strided_slice_44StridedSliceinput_1strided_slice_44/stack:output:0!strided_slice_44/stack_1:output:0!strided_slice_44/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_44
strided_slice_45/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ,       2
strided_slice_45/stack
strided_slice_45/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    -       2
strided_slice_45/stack_1
strided_slice_45/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_45/stack_2
strided_slice_45StridedSliceinput_1strided_slice_45/stack:output:0!strided_slice_45/stack_1:output:0!strided_slice_45/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_45
strided_slice_46/stackConst*
_output_shapes
:*
dtype0*!
valueB"    -       2
strided_slice_46/stack
strided_slice_46/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    .       2
strided_slice_46/stack_1
strided_slice_46/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_46/stack_2
strided_slice_46StridedSliceinput_1strided_slice_46/stack:output:0!strided_slice_46/stack_1:output:0!strided_slice_46/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_46
strided_slice_47/stackConst*
_output_shapes
:*
dtype0*!
valueB"    .       2
strided_slice_47/stack
strided_slice_47/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    /       2
strided_slice_47/stack_1
strided_slice_47/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_47/stack_2
strided_slice_47StridedSliceinput_1strided_slice_47/stack:output:0!strided_slice_47/stack_1:output:0!strided_slice_47/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_47
strided_slice_48/stackConst*
_output_shapes
:*
dtype0*!
valueB"    /       2
strided_slice_48/stack
strided_slice_48/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    0       2
strided_slice_48/stack_1
strided_slice_48/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_48/stack_2
strided_slice_48StridedSliceinput_1strided_slice_48/stack:output:0!strided_slice_48/stack_1:output:0!strided_slice_48/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_48
strided_slice_49/stackConst*
_output_shapes
:*
dtype0*!
valueB"    0       2
strided_slice_49/stack
strided_slice_49/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    1       2
strided_slice_49/stack_1
strided_slice_49/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_49/stack_2
strided_slice_49StridedSliceinput_1strided_slice_49/stack:output:0!strided_slice_49/stack_1:output:0!strided_slice_49/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_49
strided_slice_50/stackConst*
_output_shapes
:*
dtype0*!
valueB"    1       2
strided_slice_50/stack
strided_slice_50/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    2       2
strided_slice_50/stack_1
strided_slice_50/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_50/stack_2
strided_slice_50StridedSliceinput_1strided_slice_50/stack:output:0!strided_slice_50/stack_1:output:0!strided_slice_50/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_50
strided_slice_51/stackConst*
_output_shapes
:*
dtype0*!
valueB"    2       2
strided_slice_51/stack
strided_slice_51/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    3       2
strided_slice_51/stack_1
strided_slice_51/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_51/stack_2
strided_slice_51StridedSliceinput_1strided_slice_51/stack:output:0!strided_slice_51/stack_1:output:0!strided_slice_51/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_51\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis°
concatConcatV2strided_slice_2:output:0strided_slice_3:output:0strided_slice_4:output:0strided_slice_5:output:0strided_slice_6:output:0strided_slice_7:output:0strided_slice_8:output:0strided_slice_9:output:0strided_slice_10:output:0strided_slice_11:output:0strided_slice_12:output:0strided_slice_13:output:0strided_slice_14:output:0strided_slice_15:output:0strided_slice_16:output:0strided_slice_17:output:0strided_slice_18:output:0strided_slice_19:output:0strided_slice_20:output:0strided_slice_21:output:0strided_slice_22:output:0strided_slice_23:output:0strided_slice_24:output:0strided_slice_25:output:0strided_slice_26:output:0strided_slice_27:output:0strided_slice_28:output:0strided_slice_29:output:0strided_slice_30:output:0strided_slice_31:output:0strided_slice_32:output:0strided_slice_33:output:0strided_slice_34:output:0strided_slice_35:output:0strided_slice_36:output:0strided_slice_37:output:0strided_slice_38:output:0strided_slice_39:output:0strided_slice_40:output:0strided_slice_41:output:0strided_slice_42:output:0strided_slice_43:output:0strided_slice_44:output:0strided_slice_45:output:0strided_slice_46:output:0strided_slice_47:output:0strided_slice_48:output:0strided_slice_49:output:0strided_slice_50:output:0strided_slice_51:output:0concat/axis:output:0*
N2*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
concat
strided_slice_52/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_52/stack
strided_slice_52/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_52/stack_1
strided_slice_52/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_52/stack_2
strided_slice_52StridedSliceinput_1strided_slice_52/stack:output:0!strided_slice_52/stack_1:output:0!strided_slice_52/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_52
strided_slice_53/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_53/stack
strided_slice_53/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_53/stack_1
strided_slice_53/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_53/stack_2
strided_slice_53StridedSliceinput_1strided_slice_53/stack:output:0!strided_slice_53/stack_1:output:0!strided_slice_53/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_53
strided_slice_54/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_54/stack
strided_slice_54/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_54/stack_1
strided_slice_54/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_54/stack_2
strided_slice_54StridedSliceinput_1strided_slice_54/stack:output:0!strided_slice_54/stack_1:output:0!strided_slice_54/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_54
strided_slice_55/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_55/stack
strided_slice_55/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_55/stack_1
strided_slice_55/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_55/stack_2
strided_slice_55StridedSliceinput_1strided_slice_55/stack:output:0!strided_slice_55/stack_1:output:0!strided_slice_55/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_55
strided_slice_56/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_56/stack
strided_slice_56/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_56/stack_1
strided_slice_56/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_56/stack_2
strided_slice_56StridedSliceinput_1strided_slice_56/stack:output:0!strided_slice_56/stack_1:output:0!strided_slice_56/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_56
strided_slice_57/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_57/stack
strided_slice_57/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_57/stack_1
strided_slice_57/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_57/stack_2
strided_slice_57StridedSliceinput_1strided_slice_57/stack:output:0!strided_slice_57/stack_1:output:0!strided_slice_57/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_57
strided_slice_58/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_58/stack
strided_slice_58/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_58/stack_1
strided_slice_58/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_58/stack_2
strided_slice_58StridedSliceinput_1strided_slice_58/stack:output:0!strided_slice_58/stack_1:output:0!strided_slice_58/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_58
strided_slice_59/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_59/stack
strided_slice_59/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    	       2
strided_slice_59/stack_1
strided_slice_59/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_59/stack_2
strided_slice_59StridedSliceinput_1strided_slice_59/stack:output:0!strided_slice_59/stack_1:output:0!strided_slice_59/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_59
strided_slice_60/stackConst*
_output_shapes
:*
dtype0*!
valueB"    	       2
strided_slice_60/stack
strided_slice_60/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       2
strided_slice_60/stack_1
strided_slice_60/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_60/stack_2
strided_slice_60StridedSliceinput_1strided_slice_60/stack:output:0!strided_slice_60/stack_1:output:0!strided_slice_60/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_60
strided_slice_61/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       2
strided_slice_61/stack
strided_slice_61/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_61/stack_1
strided_slice_61/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_61/stack_2
strided_slice_61StridedSliceinput_1strided_slice_61/stack:output:0!strided_slice_61/stack_1:output:0!strided_slice_61/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_61
strided_slice_62/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_62/stack
strided_slice_62/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_62/stack_1
strided_slice_62/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_62/stack_2
strided_slice_62StridedSliceinput_1strided_slice_62/stack:output:0!strided_slice_62/stack_1:output:0!strided_slice_62/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_62
strided_slice_63/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_63/stack
strided_slice_63/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_63/stack_1
strided_slice_63/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_63/stack_2
strided_slice_63StridedSliceinput_1strided_slice_63/stack:output:0!strided_slice_63/stack_1:output:0!strided_slice_63/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_63
strided_slice_64/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_64/stack
strided_slice_64/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_64/stack_1
strided_slice_64/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_64/stack_2
strided_slice_64StridedSliceinput_1strided_slice_64/stack:output:0!strided_slice_64/stack_1:output:0!strided_slice_64/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_64
strided_slice_65/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_65/stack
strided_slice_65/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_65/stack_1
strided_slice_65/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_65/stack_2
strided_slice_65StridedSliceinput_1strided_slice_65/stack:output:0!strided_slice_65/stack_1:output:0!strided_slice_65/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_65
strided_slice_66/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_66/stack
strided_slice_66/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_66/stack_1
strided_slice_66/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_66/stack_2
strided_slice_66StridedSliceinput_1strided_slice_66/stack:output:0!strided_slice_66/stack_1:output:0!strided_slice_66/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_66
strided_slice_67/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_67/stack
strided_slice_67/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_67/stack_1
strided_slice_67/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_67/stack_2
strided_slice_67StridedSliceinput_1strided_slice_67/stack:output:0!strided_slice_67/stack_1:output:0!strided_slice_67/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_67
strided_slice_68/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_68/stack
strided_slice_68/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_68/stack_1
strided_slice_68/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_68/stack_2
strided_slice_68StridedSliceinput_1strided_slice_68/stack:output:0!strided_slice_68/stack_1:output:0!strided_slice_68/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_68
strided_slice_69/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_69/stack
strided_slice_69/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_69/stack_1
strided_slice_69/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_69/stack_2
strided_slice_69StridedSliceinput_1strided_slice_69/stack:output:0!strided_slice_69/stack_1:output:0!strided_slice_69/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_69
strided_slice_70/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_70/stack
strided_slice_70/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_70/stack_1
strided_slice_70/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_70/stack_2
strided_slice_70StridedSliceinput_1strided_slice_70/stack:output:0!strided_slice_70/stack_1:output:0!strided_slice_70/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_70
strided_slice_71/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_71/stack
strided_slice_71/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_71/stack_1
strided_slice_71/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_71/stack_2
strided_slice_71StridedSliceinput_1strided_slice_71/stack:output:0!strided_slice_71/stack_1:output:0!strided_slice_71/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_71
strided_slice_72/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_72/stack
strided_slice_72/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_72/stack_1
strided_slice_72/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_72/stack_2
strided_slice_72StridedSliceinput_1strided_slice_72/stack:output:0!strided_slice_72/stack_1:output:0!strided_slice_72/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_72
strided_slice_73/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_73/stack
strided_slice_73/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_73/stack_1
strided_slice_73/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_73/stack_2
strided_slice_73StridedSliceinput_1strided_slice_73/stack:output:0!strided_slice_73/stack_1:output:0!strided_slice_73/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_73
strided_slice_74/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_74/stack
strided_slice_74/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_74/stack_1
strided_slice_74/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_74/stack_2
strided_slice_74StridedSliceinput_1strided_slice_74/stack:output:0!strided_slice_74/stack_1:output:0!strided_slice_74/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_74
strided_slice_75/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_75/stack
strided_slice_75/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_75/stack_1
strided_slice_75/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_75/stack_2
strided_slice_75StridedSliceinput_1strided_slice_75/stack:output:0!strided_slice_75/stack_1:output:0!strided_slice_75/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_75
strided_slice_76/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_76/stack
strided_slice_76/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_76/stack_1
strided_slice_76/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_76/stack_2
strided_slice_76StridedSliceinput_1strided_slice_76/stack:output:0!strided_slice_76/stack_1:output:0!strided_slice_76/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_76
strided_slice_77/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_77/stack
strided_slice_77/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_77/stack_1
strided_slice_77/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_77/stack_2
strided_slice_77StridedSliceinput_1strided_slice_77/stack:output:0!strided_slice_77/stack_1:output:0!strided_slice_77/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_77
strided_slice_78/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_78/stack
strided_slice_78/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_78/stack_1
strided_slice_78/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_78/stack_2
strided_slice_78StridedSliceinput_1strided_slice_78/stack:output:0!strided_slice_78/stack_1:output:0!strided_slice_78/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_78
strided_slice_79/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_79/stack
strided_slice_79/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_79/stack_1
strided_slice_79/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_79/stack_2
strided_slice_79StridedSliceinput_1strided_slice_79/stack:output:0!strided_slice_79/stack_1:output:0!strided_slice_79/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_79
strided_slice_80/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_80/stack
strided_slice_80/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_80/stack_1
strided_slice_80/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_80/stack_2
strided_slice_80StridedSliceinput_1strided_slice_80/stack:output:0!strided_slice_80/stack_1:output:0!strided_slice_80/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_80
strided_slice_81/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_81/stack
strided_slice_81/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_81/stack_1
strided_slice_81/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_81/stack_2
strided_slice_81StridedSliceinput_1strided_slice_81/stack:output:0!strided_slice_81/stack_1:output:0!strided_slice_81/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_81
strided_slice_82/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_82/stack
strided_slice_82/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_82/stack_1
strided_slice_82/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_82/stack_2
strided_slice_82StridedSliceinput_1strided_slice_82/stack:output:0!strided_slice_82/stack_1:output:0!strided_slice_82/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_82
strided_slice_83/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_83/stack
strided_slice_83/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    !       2
strided_slice_83/stack_1
strided_slice_83/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_83/stack_2
strided_slice_83StridedSliceinput_1strided_slice_83/stack:output:0!strided_slice_83/stack_1:output:0!strided_slice_83/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_83
strided_slice_84/stackConst*
_output_shapes
:*
dtype0*!
valueB"    !       2
strided_slice_84/stack
strided_slice_84/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    "       2
strided_slice_84/stack_1
strided_slice_84/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_84/stack_2
strided_slice_84StridedSliceinput_1strided_slice_84/stack:output:0!strided_slice_84/stack_1:output:0!strided_slice_84/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_84
strided_slice_85/stackConst*
_output_shapes
:*
dtype0*!
valueB"    "       2
strided_slice_85/stack
strided_slice_85/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    #       2
strided_slice_85/stack_1
strided_slice_85/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_85/stack_2
strided_slice_85StridedSliceinput_1strided_slice_85/stack:output:0!strided_slice_85/stack_1:output:0!strided_slice_85/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_85
strided_slice_86/stackConst*
_output_shapes
:*
dtype0*!
valueB"    #       2
strided_slice_86/stack
strided_slice_86/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    $       2
strided_slice_86/stack_1
strided_slice_86/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_86/stack_2
strided_slice_86StridedSliceinput_1strided_slice_86/stack:output:0!strided_slice_86/stack_1:output:0!strided_slice_86/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_86
strided_slice_87/stackConst*
_output_shapes
:*
dtype0*!
valueB"    $       2
strided_slice_87/stack
strided_slice_87/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    %       2
strided_slice_87/stack_1
strided_slice_87/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_87/stack_2
strided_slice_87StridedSliceinput_1strided_slice_87/stack:output:0!strided_slice_87/stack_1:output:0!strided_slice_87/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_87
strided_slice_88/stackConst*
_output_shapes
:*
dtype0*!
valueB"    %       2
strided_slice_88/stack
strided_slice_88/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    &       2
strided_slice_88/stack_1
strided_slice_88/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_88/stack_2
strided_slice_88StridedSliceinput_1strided_slice_88/stack:output:0!strided_slice_88/stack_1:output:0!strided_slice_88/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_88
strided_slice_89/stackConst*
_output_shapes
:*
dtype0*!
valueB"    &       2
strided_slice_89/stack
strided_slice_89/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    '       2
strided_slice_89/stack_1
strided_slice_89/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_89/stack_2
strided_slice_89StridedSliceinput_1strided_slice_89/stack:output:0!strided_slice_89/stack_1:output:0!strided_slice_89/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_89
strided_slice_90/stackConst*
_output_shapes
:*
dtype0*!
valueB"    '       2
strided_slice_90/stack
strided_slice_90/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    (       2
strided_slice_90/stack_1
strided_slice_90/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_90/stack_2
strided_slice_90StridedSliceinput_1strided_slice_90/stack:output:0!strided_slice_90/stack_1:output:0!strided_slice_90/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_90
strided_slice_91/stackConst*
_output_shapes
:*
dtype0*!
valueB"    (       2
strided_slice_91/stack
strided_slice_91/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    )       2
strided_slice_91/stack_1
strided_slice_91/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_91/stack_2
strided_slice_91StridedSliceinput_1strided_slice_91/stack:output:0!strided_slice_91/stack_1:output:0!strided_slice_91/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_91
strided_slice_92/stackConst*
_output_shapes
:*
dtype0*!
valueB"    )       2
strided_slice_92/stack
strided_slice_92/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    *       2
strided_slice_92/stack_1
strided_slice_92/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_92/stack_2
strided_slice_92StridedSliceinput_1strided_slice_92/stack:output:0!strided_slice_92/stack_1:output:0!strided_slice_92/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_92
strided_slice_93/stackConst*
_output_shapes
:*
dtype0*!
valueB"    *       2
strided_slice_93/stack
strided_slice_93/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    +       2
strided_slice_93/stack_1
strided_slice_93/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_93/stack_2
strided_slice_93StridedSliceinput_1strided_slice_93/stack:output:0!strided_slice_93/stack_1:output:0!strided_slice_93/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_93
strided_slice_94/stackConst*
_output_shapes
:*
dtype0*!
valueB"    +       2
strided_slice_94/stack
strided_slice_94/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    ,       2
strided_slice_94/stack_1
strided_slice_94/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_94/stack_2
strided_slice_94StridedSliceinput_1strided_slice_94/stack:output:0!strided_slice_94/stack_1:output:0!strided_slice_94/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_94
strided_slice_95/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ,       2
strided_slice_95/stack
strided_slice_95/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    -       2
strided_slice_95/stack_1
strided_slice_95/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_95/stack_2
strided_slice_95StridedSliceinput_1strided_slice_95/stack:output:0!strided_slice_95/stack_1:output:0!strided_slice_95/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_95
strided_slice_96/stackConst*
_output_shapes
:*
dtype0*!
valueB"    -       2
strided_slice_96/stack
strided_slice_96/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    .       2
strided_slice_96/stack_1
strided_slice_96/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_96/stack_2
strided_slice_96StridedSliceinput_1strided_slice_96/stack:output:0!strided_slice_96/stack_1:output:0!strided_slice_96/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_96
strided_slice_97/stackConst*
_output_shapes
:*
dtype0*!
valueB"    .       2
strided_slice_97/stack
strided_slice_97/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    /       2
strided_slice_97/stack_1
strided_slice_97/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_97/stack_2
strided_slice_97StridedSliceinput_1strided_slice_97/stack:output:0!strided_slice_97/stack_1:output:0!strided_slice_97/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_97
strided_slice_98/stackConst*
_output_shapes
:*
dtype0*!
valueB"    /       2
strided_slice_98/stack
strided_slice_98/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    0       2
strided_slice_98/stack_1
strided_slice_98/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_98/stack_2
strided_slice_98StridedSliceinput_1strided_slice_98/stack:output:0!strided_slice_98/stack_1:output:0!strided_slice_98/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_98
strided_slice_99/stackConst*
_output_shapes
:*
dtype0*!
valueB"    0       2
strided_slice_99/stack
strided_slice_99/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    1       2
strided_slice_99/stack_1
strided_slice_99/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_99/stack_2
strided_slice_99StridedSliceinput_1strided_slice_99/stack:output:0!strided_slice_99/stack_1:output:0!strided_slice_99/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_99
strided_slice_100/stackConst*
_output_shapes
:*
dtype0*!
valueB"    1       2
strided_slice_100/stack
strided_slice_100/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    2       2
strided_slice_100/stack_1
strided_slice_100/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_100/stack_2
strided_slice_100StridedSliceinput_1 strided_slice_100/stack:output:0"strided_slice_100/stack_1:output:0"strided_slice_100/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_100
strided_slice_101/stackConst*
_output_shapes
:*
dtype0*!
valueB"    2       2
strided_slice_101/stack
strided_slice_101/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    3       2
strided_slice_101/stack_1
strided_slice_101/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_101/stack_2
strided_slice_101StridedSliceinput_1 strided_slice_101/stack:output:0"strided_slice_101/stack_1:output:0"strided_slice_101/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask2
strided_slice_101`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axisÀ
concat_1ConcatV2strided_slice_52:output:0strided_slice_53:output:0strided_slice_54:output:0strided_slice_55:output:0strided_slice_56:output:0strided_slice_57:output:0strided_slice_58:output:0strided_slice_59:output:0strided_slice_60:output:0strided_slice_61:output:0strided_slice_62:output:0strided_slice_63:output:0strided_slice_64:output:0strided_slice_65:output:0strided_slice_66:output:0strided_slice_67:output:0strided_slice_68:output:0strided_slice_69:output:0strided_slice_70:output:0strided_slice_71:output:0strided_slice_72:output:0strided_slice_73:output:0strided_slice_74:output:0strided_slice_75:output:0strided_slice_76:output:0strided_slice_77:output:0strided_slice_78:output:0strided_slice_79:output:0strided_slice_80:output:0strided_slice_81:output:0strided_slice_82:output:0strided_slice_83:output:0strided_slice_84:output:0strided_slice_85:output:0strided_slice_86:output:0strided_slice_87:output:0strided_slice_88:output:0strided_slice_89:output:0strided_slice_90:output:0strided_slice_91:output:0strided_slice_92:output:0strided_slice_93:output:0strided_slice_94:output:0strided_slice_95:output:0strided_slice_96:output:0strided_slice_97:output:0strided_slice_98:output:0strided_slice_99:output:0strided_slice_100:output:0strided_slice_101:output:0concat_1/axis:output:0*
N2*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

concat_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape/shapeq
ReshapeReshapeinput_1Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_1/shape
	Reshape_1Reshapestrided_slice:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_2/shape
	Reshape_2Reshapestrided_slice_1:output:0Reshape_2/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_2s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_3/shape
	Reshape_3Reshapeconcat_1:output:0Reshape_3/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_3£
&conv_res_block/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv_res_block_3112520conv_res_block_3112522conv_res_block_3112524conv_res_block_3112526conv_res_block_3112528conv_res_block_3112530conv_res_block_3112532conv_res_block_3112534conv_res_block_3112536conv_res_block_3112538conv_res_block_3112540conv_res_block_3112542*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*U
fPRN
L__inference_conv_res_block_layer_call_and_return_conditional_losses_134691662(
&conv_res_block/StatefulPartitionedCallË
conv_res_block/IdentityIdentity/conv_res_block/StatefulPartitionedCall:output:0'^conv_res_block/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block/Identity
%inner_encoder/StatefulPartitionedCallStatefulPartitionedCall conv_res_block/Identity:output:0inner_encoder_3112570*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_encoder_layer_call_and_return_conditional_losses_134702002'
%inner_encoder/StatefulPartitionedCallÆ
inner_encoder/IdentityIdentity.inner_encoder/StatefulPartitionedCall:output:0&^inner_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_encoder/Identity
%inner_decoder/StatefulPartitionedCallStatefulPartitionedCallinner_encoder/Identity:output:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772'
%inner_decoder/StatefulPartitionedCallÇ
inner_decoder/IdentityIdentity.inner_decoder/StatefulPartitionedCall:output:0&^inner_decoder/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder/IdentityÐ
(conv_res_block_1/StatefulPartitionedCallStatefulPartitionedCallinner_decoder/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372*
(conv_res_block_1/StatefulPartitionedCallÓ
conv_res_block_1/IdentityIdentity1conv_res_block_1/StatefulPartitionedCall:output:0)^conv_res_block_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1/Identityw
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ3      2
Reshape_4/shape
	Reshape_4Reshape"conv_res_block_1/Identity:output:0Reshape_4/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
	Reshape_4Õ
*conv_res_block_1_1/StatefulPartitionedCallStatefulPartitionedCall conv_res_block/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372,
*conv_res_block_1_1/StatefulPartitionedCallÛ
conv_res_block_1_1/IdentityIdentity3conv_res_block_1_1/StatefulPartitionedCall:output:0+^conv_res_block_1_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_1/Identityw
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ3      2
Reshape_5/shape
	Reshape_5Reshape$conv_res_block_1_1/Identity:output:0Reshape_5/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
	Reshape_5

RelMSE/subSubinner_decoder/Identity:output:0 conv_res_block/Identity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

RelMSE/subk
RelMSE/SquareSquareRelMSE/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
RelMSE/Square
RelMSE/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
RelMSE/Mean/reduction_indices
RelMSE/MeanMeanRelMSE/Square:y:0&RelMSE/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
RelMSE/Mean
RelMSE/Square_1Square conv_res_block/Identity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
RelMSE/Square_1
RelMSE/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
RelMSE/Mean_1/reduction_indices
RelMSE/Mean_1MeanRelMSE/Square_1:y:0(RelMSE/Mean_1/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
RelMSE/Mean_1a
RelMSE/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'72
RelMSE/add/y~

RelMSE/addAddV2RelMSE/Mean_1:output:0RelMSE/add/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

RelMSE/add
RelMSE/truedivRealDivRelMSE/Mean:output:0RelMSE/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
RelMSE/truediv
RelMSE/Mean_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
RelMSE/Mean_2/reduction_indices
RelMSE/Mean_2MeanRelMSE/truediv:z:0(RelMSE/Mean_2/reduction_indices:output:0*
T0*
_output_shapes
: 2
RelMSE/Mean_2
RelMSE/weighted_loss/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
RelMSE/weighted_loss/Cast/x
RelMSE/weighted_loss/MulMulRelMSE/Mean_2:output:0$RelMSE/weighted_loss/Cast/x:output:0*
T0*
_output_shapes
: 2
RelMSE/weighted_loss/Mul{
RelMSE/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
RelMSE/weighted_loss/Const
RelMSE/weighted_loss/SumSumRelMSE/weighted_loss/Mul:z:0#RelMSE/weighted_loss/Const:output:0*
T0*
_output_shapes
: 2
RelMSE/weighted_loss/Sum
!RelMSE/weighted_loss/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :2#
!RelMSE/weighted_loss/num_elements´
&RelMSE/weighted_loss/num_elements/CastCast*RelMSE/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: 2(
&RelMSE/weighted_loss/num_elements/Cast
RelMSE/weighted_loss/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
RelMSE/weighted_loss/Const_1ª
RelMSE/weighted_loss/Sum_1Sum!RelMSE/weighted_loss/Sum:output:0%RelMSE/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 2
RelMSE/weighted_loss/Sum_1¶
RelMSE/weighted_loss/valueDivNoNan#RelMSE/weighted_loss/Sum_1:output:0*RelMSE/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2
RelMSE/weighted_loss/valueS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
mul/xb
mulMulmul/x:output:0RelMSE/weighted_loss/value:z:0*
T0*
_output_shapes
: 2
mul©
(conv_res_block_2/StatefulPartitionedCallStatefulPartitionedCallReshape_1:output:0conv_res_block_3112520conv_res_block_3112522conv_res_block_3112524conv_res_block_3112526conv_res_block_3112528conv_res_block_3112530conv_res_block_3112532conv_res_block_3112534conv_res_block_3112536conv_res_block_3112538conv_res_block_3112540conv_res_block_3112542*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*U
fPRN
L__inference_conv_res_block_layer_call_and_return_conditional_losses_134691662*
(conv_res_block_2/StatefulPartitionedCallÓ
conv_res_block_2/IdentityIdentity1conv_res_block_2/StatefulPartitionedCall:output:0)^conv_res_block_2/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_2/Identity
'inner_encoder_1/StatefulPartitionedCallStatefulPartitionedCall"conv_res_block_2/Identity:output:0inner_encoder_3112570*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_encoder_layer_call_and_return_conditional_losses_134702002)
'inner_encoder_1/StatefulPartitionedCallÎ
inner_encoder_1/IdentityIdentity0inner_encoder_1/StatefulPartitionedCall:output:0(^inner_encoder_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_encoder_1/Identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOp
MatMulMatMul!inner_encoder_1/Identity:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
'inner_decoder_1/StatefulPartitionedCallStatefulPartitionedCallMatMul:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772)
'inner_decoder_1/StatefulPartitionedCallÏ
inner_decoder_1/IdentityIdentity0inner_decoder_1/StatefulPartitionedCall:output:0(^inner_decoder_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_1/IdentityÖ
*conv_res_block_1_2/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_1/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372,
*conv_res_block_1_2/StatefulPartitionedCallÛ
conv_res_block_1_2/IdentityIdentity3conv_res_block_1_2/StatefulPartitionedCall:output:0+^conv_res_block_1_2/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_2/Identityw
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_6/shape
	Reshape_6Reshape$conv_res_block_1_2/Identity:output:0Reshape_6/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_6d
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
MatMul_1/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulIdentity:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1
'inner_decoder_2/StatefulPartitionedCallStatefulPartitionedCallMatMul_1:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772)
'inner_decoder_2/StatefulPartitionedCallÏ
inner_decoder_2/IdentityIdentity0inner_decoder_2/StatefulPartitionedCall:output:0(^inner_decoder_2/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_2/IdentityÖ
*conv_res_block_1_3/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_2/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372,
*conv_res_block_1_3/StatefulPartitionedCallÛ
conv_res_block_1_3/IdentityIdentity3conv_res_block_1_3/StatefulPartitionedCall:output:0+^conv_res_block_1_3/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_3/Identityw
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_7/shape
	Reshape_7Reshape$conv_res_block_1_3/Identity:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_7j

Identity_1IdentityMatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1
MatMul_2/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp
MatMul_2MatMulIdentity_1:output:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_2
'inner_decoder_3/StatefulPartitionedCallStatefulPartitionedCallMatMul_2:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772)
'inner_decoder_3/StatefulPartitionedCallÏ
inner_decoder_3/IdentityIdentity0inner_decoder_3/StatefulPartitionedCall:output:0(^inner_decoder_3/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_3/IdentityÖ
*conv_res_block_1_4/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_3/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372,
*conv_res_block_1_4/StatefulPartitionedCallÛ
conv_res_block_1_4/IdentityIdentity3conv_res_block_1_4/StatefulPartitionedCall:output:0+^conv_res_block_1_4/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_4/Identityw
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_8/shape
	Reshape_8Reshape$conv_res_block_1_4/Identity:output:0Reshape_8/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_8j

Identity_2IdentityMatMul_2:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
MatMul_3/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_3/ReadVariableOp
MatMul_3MatMulIdentity_2:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_3
'inner_decoder_4/StatefulPartitionedCallStatefulPartitionedCallMatMul_3:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772)
'inner_decoder_4/StatefulPartitionedCallÏ
inner_decoder_4/IdentityIdentity0inner_decoder_4/StatefulPartitionedCall:output:0(^inner_decoder_4/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_4/IdentityÖ
*conv_res_block_1_5/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_4/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372,
*conv_res_block_1_5/StatefulPartitionedCallÛ
conv_res_block_1_5/IdentityIdentity3conv_res_block_1_5/StatefulPartitionedCall:output:0+^conv_res_block_1_5/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_5/Identityw
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_9/shape
	Reshape_9Reshape$conv_res_block_1_5/Identity:output:0Reshape_9/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_9j

Identity_3IdentityMatMul_3:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3
MatMul_4/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_4/ReadVariableOp
MatMul_4MatMulIdentity_3:output:0MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_4
'inner_decoder_5/StatefulPartitionedCallStatefulPartitionedCallMatMul_4:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772)
'inner_decoder_5/StatefulPartitionedCallÏ
inner_decoder_5/IdentityIdentity0inner_decoder_5/StatefulPartitionedCall:output:0(^inner_decoder_5/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_5/IdentityÖ
*conv_res_block_1_6/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_5/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372,
*conv_res_block_1_6/StatefulPartitionedCallÛ
conv_res_block_1_6/IdentityIdentity3conv_res_block_1_6/StatefulPartitionedCall:output:0+^conv_res_block_1_6/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_6/Identityy
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_10/shape

Reshape_10Reshape$conv_res_block_1_6/Identity:output:0Reshape_10/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_10j

Identity_4IdentityMatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_4
MatMul_5/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_5/ReadVariableOp
MatMul_5MatMulIdentity_4:output:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_5
'inner_decoder_6/StatefulPartitionedCallStatefulPartitionedCallMatMul_5:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772)
'inner_decoder_6/StatefulPartitionedCallÏ
inner_decoder_6/IdentityIdentity0inner_decoder_6/StatefulPartitionedCall:output:0(^inner_decoder_6/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_6/IdentityÖ
*conv_res_block_1_7/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_6/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372,
*conv_res_block_1_7/StatefulPartitionedCallÛ
conv_res_block_1_7/IdentityIdentity3conv_res_block_1_7/StatefulPartitionedCall:output:0+^conv_res_block_1_7/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_7/Identityy
Reshape_11/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_11/shape

Reshape_11Reshape$conv_res_block_1_7/Identity:output:0Reshape_11/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_11j

Identity_5IdentityMatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_5
MatMul_6/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_6/ReadVariableOp
MatMul_6MatMulIdentity_5:output:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_6
'inner_decoder_7/StatefulPartitionedCallStatefulPartitionedCallMatMul_6:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772)
'inner_decoder_7/StatefulPartitionedCallÏ
inner_decoder_7/IdentityIdentity0inner_decoder_7/StatefulPartitionedCall:output:0(^inner_decoder_7/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_7/IdentityÖ
*conv_res_block_1_8/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_7/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372,
*conv_res_block_1_8/StatefulPartitionedCallÛ
conv_res_block_1_8/IdentityIdentity3conv_res_block_1_8/StatefulPartitionedCall:output:0+^conv_res_block_1_8/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_8/Identityy
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_12/shape

Reshape_12Reshape$conv_res_block_1_8/Identity:output:0Reshape_12/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_12j

Identity_6IdentityMatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_6
MatMul_7/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_7/ReadVariableOp
MatMul_7MatMulIdentity_6:output:0MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_7
'inner_decoder_8/StatefulPartitionedCallStatefulPartitionedCallMatMul_7:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772)
'inner_decoder_8/StatefulPartitionedCallÏ
inner_decoder_8/IdentityIdentity0inner_decoder_8/StatefulPartitionedCall:output:0(^inner_decoder_8/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_8/IdentityÖ
*conv_res_block_1_9/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_8/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372,
*conv_res_block_1_9/StatefulPartitionedCallÛ
conv_res_block_1_9/IdentityIdentity3conv_res_block_1_9/StatefulPartitionedCall:output:0+^conv_res_block_1_9/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_9/Identityy
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_13/shape

Reshape_13Reshape$conv_res_block_1_9/Identity:output:0Reshape_13/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_13j

Identity_7IdentityMatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_7
MatMul_8/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_8/ReadVariableOp
MatMul_8MatMulIdentity_7:output:0MatMul_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_8
'inner_decoder_9/StatefulPartitionedCallStatefulPartitionedCallMatMul_8:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772)
'inner_decoder_9/StatefulPartitionedCallÏ
inner_decoder_9/IdentityIdentity0inner_decoder_9/StatefulPartitionedCall:output:0(^inner_decoder_9/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_9/IdentityØ
+conv_res_block_1_10/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_9/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_10/StatefulPartitionedCallß
conv_res_block_1_10/IdentityIdentity4conv_res_block_1_10/StatefulPartitionedCall:output:0,^conv_res_block_1_10/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_10/Identityy
Reshape_14/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_14/shape

Reshape_14Reshape%conv_res_block_1_10/Identity:output:0Reshape_14/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_14j

Identity_8IdentityMatMul_8:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_8
MatMul_9/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_9/ReadVariableOp
MatMul_9MatMulIdentity_8:output:0MatMul_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_9
(inner_decoder_10/StatefulPartitionedCallStatefulPartitionedCallMatMul_9:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_10/StatefulPartitionedCallÓ
inner_decoder_10/IdentityIdentity1inner_decoder_10/StatefulPartitionedCall:output:0)^inner_decoder_10/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_10/IdentityÙ
+conv_res_block_1_11/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_10/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_11/StatefulPartitionedCallß
conv_res_block_1_11/IdentityIdentity4conv_res_block_1_11/StatefulPartitionedCall:output:0,^conv_res_block_1_11/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_11/Identityy
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_15/shape

Reshape_15Reshape%conv_res_block_1_11/Identity:output:0Reshape_15/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_15j

Identity_9IdentityMatMul_9:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_9
MatMul_10/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_10/ReadVariableOp
	MatMul_10MatMulIdentity_9:output:0 MatMul_10/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_10
(inner_decoder_11/StatefulPartitionedCallStatefulPartitionedCallMatMul_10:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_11/StatefulPartitionedCallÓ
inner_decoder_11/IdentityIdentity1inner_decoder_11/StatefulPartitionedCall:output:0)^inner_decoder_11/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_11/IdentityÙ
+conv_res_block_1_12/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_11/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_12/StatefulPartitionedCallß
conv_res_block_1_12/IdentityIdentity4conv_res_block_1_12/StatefulPartitionedCall:output:0,^conv_res_block_1_12/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_12/Identityy
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_16/shape

Reshape_16Reshape%conv_res_block_1_12/Identity:output:0Reshape_16/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_16m
Identity_10IdentityMatMul_10:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_10
MatMul_11/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_11/ReadVariableOp
	MatMul_11MatMulIdentity_10:output:0 MatMul_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_11
(inner_decoder_12/StatefulPartitionedCallStatefulPartitionedCallMatMul_11:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_12/StatefulPartitionedCallÓ
inner_decoder_12/IdentityIdentity1inner_decoder_12/StatefulPartitionedCall:output:0)^inner_decoder_12/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_12/IdentityÙ
+conv_res_block_1_13/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_12/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_13/StatefulPartitionedCallß
conv_res_block_1_13/IdentityIdentity4conv_res_block_1_13/StatefulPartitionedCall:output:0,^conv_res_block_1_13/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_13/Identityy
Reshape_17/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_17/shape

Reshape_17Reshape%conv_res_block_1_13/Identity:output:0Reshape_17/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_17m
Identity_11IdentityMatMul_11:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_11
MatMul_12/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_12/ReadVariableOp
	MatMul_12MatMulIdentity_11:output:0 MatMul_12/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_12
(inner_decoder_13/StatefulPartitionedCallStatefulPartitionedCallMatMul_12:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_13/StatefulPartitionedCallÓ
inner_decoder_13/IdentityIdentity1inner_decoder_13/StatefulPartitionedCall:output:0)^inner_decoder_13/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_13/IdentityÙ
+conv_res_block_1_14/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_13/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_14/StatefulPartitionedCallß
conv_res_block_1_14/IdentityIdentity4conv_res_block_1_14/StatefulPartitionedCall:output:0,^conv_res_block_1_14/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_14/Identityy
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_18/shape

Reshape_18Reshape%conv_res_block_1_14/Identity:output:0Reshape_18/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_18m
Identity_12IdentityMatMul_12:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_12
MatMul_13/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_13/ReadVariableOp
	MatMul_13MatMulIdentity_12:output:0 MatMul_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_13
(inner_decoder_14/StatefulPartitionedCallStatefulPartitionedCallMatMul_13:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_14/StatefulPartitionedCallÓ
inner_decoder_14/IdentityIdentity1inner_decoder_14/StatefulPartitionedCall:output:0)^inner_decoder_14/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_14/IdentityÙ
+conv_res_block_1_15/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_14/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_15/StatefulPartitionedCallß
conv_res_block_1_15/IdentityIdentity4conv_res_block_1_15/StatefulPartitionedCall:output:0,^conv_res_block_1_15/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_15/Identityy
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_19/shape

Reshape_19Reshape%conv_res_block_1_15/Identity:output:0Reshape_19/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_19m
Identity_13IdentityMatMul_13:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_13
MatMul_14/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_14/ReadVariableOp
	MatMul_14MatMulIdentity_13:output:0 MatMul_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_14
(inner_decoder_15/StatefulPartitionedCallStatefulPartitionedCallMatMul_14:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_15/StatefulPartitionedCallÓ
inner_decoder_15/IdentityIdentity1inner_decoder_15/StatefulPartitionedCall:output:0)^inner_decoder_15/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_15/IdentityÙ
+conv_res_block_1_16/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_15/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_16/StatefulPartitionedCallß
conv_res_block_1_16/IdentityIdentity4conv_res_block_1_16/StatefulPartitionedCall:output:0,^conv_res_block_1_16/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_16/Identityy
Reshape_20/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_20/shape

Reshape_20Reshape%conv_res_block_1_16/Identity:output:0Reshape_20/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_20m
Identity_14IdentityMatMul_14:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_14
MatMul_15/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_15/ReadVariableOp
	MatMul_15MatMulIdentity_14:output:0 MatMul_15/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_15
(inner_decoder_16/StatefulPartitionedCallStatefulPartitionedCallMatMul_15:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_16/StatefulPartitionedCallÓ
inner_decoder_16/IdentityIdentity1inner_decoder_16/StatefulPartitionedCall:output:0)^inner_decoder_16/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_16/IdentityÙ
+conv_res_block_1_17/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_16/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_17/StatefulPartitionedCallß
conv_res_block_1_17/IdentityIdentity4conv_res_block_1_17/StatefulPartitionedCall:output:0,^conv_res_block_1_17/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_17/Identityy
Reshape_21/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_21/shape

Reshape_21Reshape%conv_res_block_1_17/Identity:output:0Reshape_21/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_21m
Identity_15IdentityMatMul_15:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_15
MatMul_16/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_16/ReadVariableOp
	MatMul_16MatMulIdentity_15:output:0 MatMul_16/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_16
(inner_decoder_17/StatefulPartitionedCallStatefulPartitionedCallMatMul_16:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_17/StatefulPartitionedCallÓ
inner_decoder_17/IdentityIdentity1inner_decoder_17/StatefulPartitionedCall:output:0)^inner_decoder_17/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_17/IdentityÙ
+conv_res_block_1_18/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_17/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_18/StatefulPartitionedCallß
conv_res_block_1_18/IdentityIdentity4conv_res_block_1_18/StatefulPartitionedCall:output:0,^conv_res_block_1_18/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_18/Identityy
Reshape_22/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_22/shape

Reshape_22Reshape%conv_res_block_1_18/Identity:output:0Reshape_22/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_22m
Identity_16IdentityMatMul_16:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_16
MatMul_17/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_17/ReadVariableOp
	MatMul_17MatMulIdentity_16:output:0 MatMul_17/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_17
(inner_decoder_18/StatefulPartitionedCallStatefulPartitionedCallMatMul_17:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_18/StatefulPartitionedCallÓ
inner_decoder_18/IdentityIdentity1inner_decoder_18/StatefulPartitionedCall:output:0)^inner_decoder_18/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_18/IdentityÙ
+conv_res_block_1_19/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_18/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_19/StatefulPartitionedCallß
conv_res_block_1_19/IdentityIdentity4conv_res_block_1_19/StatefulPartitionedCall:output:0,^conv_res_block_1_19/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_19/Identityy
Reshape_23/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_23/shape

Reshape_23Reshape%conv_res_block_1_19/Identity:output:0Reshape_23/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_23m
Identity_17IdentityMatMul_17:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_17
MatMul_18/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_18/ReadVariableOp
	MatMul_18MatMulIdentity_17:output:0 MatMul_18/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_18
(inner_decoder_19/StatefulPartitionedCallStatefulPartitionedCallMatMul_18:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_19/StatefulPartitionedCallÓ
inner_decoder_19/IdentityIdentity1inner_decoder_19/StatefulPartitionedCall:output:0)^inner_decoder_19/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_19/IdentityÙ
+conv_res_block_1_20/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_19/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_20/StatefulPartitionedCallß
conv_res_block_1_20/IdentityIdentity4conv_res_block_1_20/StatefulPartitionedCall:output:0,^conv_res_block_1_20/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_20/Identityy
Reshape_24/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_24/shape

Reshape_24Reshape%conv_res_block_1_20/Identity:output:0Reshape_24/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_24m
Identity_18IdentityMatMul_18:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_18
MatMul_19/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_19/ReadVariableOp
	MatMul_19MatMulIdentity_18:output:0 MatMul_19/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_19
(inner_decoder_20/StatefulPartitionedCallStatefulPartitionedCallMatMul_19:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_20/StatefulPartitionedCallÓ
inner_decoder_20/IdentityIdentity1inner_decoder_20/StatefulPartitionedCall:output:0)^inner_decoder_20/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_20/IdentityÙ
+conv_res_block_1_21/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_20/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_21/StatefulPartitionedCallß
conv_res_block_1_21/IdentityIdentity4conv_res_block_1_21/StatefulPartitionedCall:output:0,^conv_res_block_1_21/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_21/Identityy
Reshape_25/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_25/shape

Reshape_25Reshape%conv_res_block_1_21/Identity:output:0Reshape_25/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_25m
Identity_19IdentityMatMul_19:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_19
MatMul_20/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_20/ReadVariableOp
	MatMul_20MatMulIdentity_19:output:0 MatMul_20/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_20
(inner_decoder_21/StatefulPartitionedCallStatefulPartitionedCallMatMul_20:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_21/StatefulPartitionedCallÓ
inner_decoder_21/IdentityIdentity1inner_decoder_21/StatefulPartitionedCall:output:0)^inner_decoder_21/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_21/IdentityÙ
+conv_res_block_1_22/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_21/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_22/StatefulPartitionedCallß
conv_res_block_1_22/IdentityIdentity4conv_res_block_1_22/StatefulPartitionedCall:output:0,^conv_res_block_1_22/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_22/Identityy
Reshape_26/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_26/shape

Reshape_26Reshape%conv_res_block_1_22/Identity:output:0Reshape_26/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_26m
Identity_20IdentityMatMul_20:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_20
MatMul_21/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_21/ReadVariableOp
	MatMul_21MatMulIdentity_20:output:0 MatMul_21/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_21
(inner_decoder_22/StatefulPartitionedCallStatefulPartitionedCallMatMul_21:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_22/StatefulPartitionedCallÓ
inner_decoder_22/IdentityIdentity1inner_decoder_22/StatefulPartitionedCall:output:0)^inner_decoder_22/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_22/IdentityÙ
+conv_res_block_1_23/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_22/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_23/StatefulPartitionedCallß
conv_res_block_1_23/IdentityIdentity4conv_res_block_1_23/StatefulPartitionedCall:output:0,^conv_res_block_1_23/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_23/Identityy
Reshape_27/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_27/shape

Reshape_27Reshape%conv_res_block_1_23/Identity:output:0Reshape_27/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_27m
Identity_21IdentityMatMul_21:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_21
MatMul_22/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_22/ReadVariableOp
	MatMul_22MatMulIdentity_21:output:0 MatMul_22/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_22
(inner_decoder_23/StatefulPartitionedCallStatefulPartitionedCallMatMul_22:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_23/StatefulPartitionedCallÓ
inner_decoder_23/IdentityIdentity1inner_decoder_23/StatefulPartitionedCall:output:0)^inner_decoder_23/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_23/IdentityÙ
+conv_res_block_1_24/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_23/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_24/StatefulPartitionedCallß
conv_res_block_1_24/IdentityIdentity4conv_res_block_1_24/StatefulPartitionedCall:output:0,^conv_res_block_1_24/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_24/Identityy
Reshape_28/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_28/shape

Reshape_28Reshape%conv_res_block_1_24/Identity:output:0Reshape_28/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_28m
Identity_22IdentityMatMul_22:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_22
MatMul_23/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_23/ReadVariableOp
	MatMul_23MatMulIdentity_22:output:0 MatMul_23/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_23
(inner_decoder_24/StatefulPartitionedCallStatefulPartitionedCallMatMul_23:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_24/StatefulPartitionedCallÓ
inner_decoder_24/IdentityIdentity1inner_decoder_24/StatefulPartitionedCall:output:0)^inner_decoder_24/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_24/IdentityÙ
+conv_res_block_1_25/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_24/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_25/StatefulPartitionedCallß
conv_res_block_1_25/IdentityIdentity4conv_res_block_1_25/StatefulPartitionedCall:output:0,^conv_res_block_1_25/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_25/Identityy
Reshape_29/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_29/shape

Reshape_29Reshape%conv_res_block_1_25/Identity:output:0Reshape_29/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_29m
Identity_23IdentityMatMul_23:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_23
MatMul_24/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_24/ReadVariableOp
	MatMul_24MatMulIdentity_23:output:0 MatMul_24/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_24
(inner_decoder_25/StatefulPartitionedCallStatefulPartitionedCallMatMul_24:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_25/StatefulPartitionedCallÓ
inner_decoder_25/IdentityIdentity1inner_decoder_25/StatefulPartitionedCall:output:0)^inner_decoder_25/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_25/IdentityÙ
+conv_res_block_1_26/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_25/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_26/StatefulPartitionedCallß
conv_res_block_1_26/IdentityIdentity4conv_res_block_1_26/StatefulPartitionedCall:output:0,^conv_res_block_1_26/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_26/Identityy
Reshape_30/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_30/shape

Reshape_30Reshape%conv_res_block_1_26/Identity:output:0Reshape_30/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_30m
Identity_24IdentityMatMul_24:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_24
MatMul_25/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_25/ReadVariableOp
	MatMul_25MatMulIdentity_24:output:0 MatMul_25/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_25
(inner_decoder_26/StatefulPartitionedCallStatefulPartitionedCallMatMul_25:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_26/StatefulPartitionedCallÓ
inner_decoder_26/IdentityIdentity1inner_decoder_26/StatefulPartitionedCall:output:0)^inner_decoder_26/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_26/IdentityÙ
+conv_res_block_1_27/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_26/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_27/StatefulPartitionedCallß
conv_res_block_1_27/IdentityIdentity4conv_res_block_1_27/StatefulPartitionedCall:output:0,^conv_res_block_1_27/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_27/Identityy
Reshape_31/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_31/shape

Reshape_31Reshape%conv_res_block_1_27/Identity:output:0Reshape_31/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_31m
Identity_25IdentityMatMul_25:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_25
MatMul_26/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_26/ReadVariableOp
	MatMul_26MatMulIdentity_25:output:0 MatMul_26/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_26
(inner_decoder_27/StatefulPartitionedCallStatefulPartitionedCallMatMul_26:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_27/StatefulPartitionedCallÓ
inner_decoder_27/IdentityIdentity1inner_decoder_27/StatefulPartitionedCall:output:0)^inner_decoder_27/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_27/IdentityÙ
+conv_res_block_1_28/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_27/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_28/StatefulPartitionedCallß
conv_res_block_1_28/IdentityIdentity4conv_res_block_1_28/StatefulPartitionedCall:output:0,^conv_res_block_1_28/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_28/Identityy
Reshape_32/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_32/shape

Reshape_32Reshape%conv_res_block_1_28/Identity:output:0Reshape_32/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_32m
Identity_26IdentityMatMul_26:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_26
MatMul_27/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_27/ReadVariableOp
	MatMul_27MatMulIdentity_26:output:0 MatMul_27/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_27
(inner_decoder_28/StatefulPartitionedCallStatefulPartitionedCallMatMul_27:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_28/StatefulPartitionedCallÓ
inner_decoder_28/IdentityIdentity1inner_decoder_28/StatefulPartitionedCall:output:0)^inner_decoder_28/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_28/IdentityÙ
+conv_res_block_1_29/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_28/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_29/StatefulPartitionedCallß
conv_res_block_1_29/IdentityIdentity4conv_res_block_1_29/StatefulPartitionedCall:output:0,^conv_res_block_1_29/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_29/Identityy
Reshape_33/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_33/shape

Reshape_33Reshape%conv_res_block_1_29/Identity:output:0Reshape_33/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_33m
Identity_27IdentityMatMul_27:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_27
MatMul_28/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_28/ReadVariableOp
	MatMul_28MatMulIdentity_27:output:0 MatMul_28/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_28
(inner_decoder_29/StatefulPartitionedCallStatefulPartitionedCallMatMul_28:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_29/StatefulPartitionedCallÓ
inner_decoder_29/IdentityIdentity1inner_decoder_29/StatefulPartitionedCall:output:0)^inner_decoder_29/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_29/IdentityÙ
+conv_res_block_1_30/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_29/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_30/StatefulPartitionedCallß
conv_res_block_1_30/IdentityIdentity4conv_res_block_1_30/StatefulPartitionedCall:output:0,^conv_res_block_1_30/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_30/Identityy
Reshape_34/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_34/shape

Reshape_34Reshape%conv_res_block_1_30/Identity:output:0Reshape_34/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_34m
Identity_28IdentityMatMul_28:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_28
MatMul_29/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_29/ReadVariableOp
	MatMul_29MatMulIdentity_28:output:0 MatMul_29/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_29
(inner_decoder_30/StatefulPartitionedCallStatefulPartitionedCallMatMul_29:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_30/StatefulPartitionedCallÓ
inner_decoder_30/IdentityIdentity1inner_decoder_30/StatefulPartitionedCall:output:0)^inner_decoder_30/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_30/IdentityÙ
+conv_res_block_1_31/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_30/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_31/StatefulPartitionedCallß
conv_res_block_1_31/IdentityIdentity4conv_res_block_1_31/StatefulPartitionedCall:output:0,^conv_res_block_1_31/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_31/Identityy
Reshape_35/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_35/shape

Reshape_35Reshape%conv_res_block_1_31/Identity:output:0Reshape_35/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_35m
Identity_29IdentityMatMul_29:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_29
MatMul_30/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_30/ReadVariableOp
	MatMul_30MatMulIdentity_29:output:0 MatMul_30/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_30
(inner_decoder_31/StatefulPartitionedCallStatefulPartitionedCallMatMul_30:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_31/StatefulPartitionedCallÓ
inner_decoder_31/IdentityIdentity1inner_decoder_31/StatefulPartitionedCall:output:0)^inner_decoder_31/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_31/IdentityÙ
+conv_res_block_1_32/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_31/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_32/StatefulPartitionedCallß
conv_res_block_1_32/IdentityIdentity4conv_res_block_1_32/StatefulPartitionedCall:output:0,^conv_res_block_1_32/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_32/Identityy
Reshape_36/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_36/shape

Reshape_36Reshape%conv_res_block_1_32/Identity:output:0Reshape_36/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_36m
Identity_30IdentityMatMul_30:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_30
MatMul_31/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_31/ReadVariableOp
	MatMul_31MatMulIdentity_30:output:0 MatMul_31/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_31
(inner_decoder_32/StatefulPartitionedCallStatefulPartitionedCallMatMul_31:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_32/StatefulPartitionedCallÓ
inner_decoder_32/IdentityIdentity1inner_decoder_32/StatefulPartitionedCall:output:0)^inner_decoder_32/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_32/IdentityÙ
+conv_res_block_1_33/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_32/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_33/StatefulPartitionedCallß
conv_res_block_1_33/IdentityIdentity4conv_res_block_1_33/StatefulPartitionedCall:output:0,^conv_res_block_1_33/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_33/Identityy
Reshape_37/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_37/shape

Reshape_37Reshape%conv_res_block_1_33/Identity:output:0Reshape_37/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_37m
Identity_31IdentityMatMul_31:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_31
MatMul_32/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_32/ReadVariableOp
	MatMul_32MatMulIdentity_31:output:0 MatMul_32/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_32
(inner_decoder_33/StatefulPartitionedCallStatefulPartitionedCallMatMul_32:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_33/StatefulPartitionedCallÓ
inner_decoder_33/IdentityIdentity1inner_decoder_33/StatefulPartitionedCall:output:0)^inner_decoder_33/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_33/IdentityÙ
+conv_res_block_1_34/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_33/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_34/StatefulPartitionedCallß
conv_res_block_1_34/IdentityIdentity4conv_res_block_1_34/StatefulPartitionedCall:output:0,^conv_res_block_1_34/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_34/Identityy
Reshape_38/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_38/shape

Reshape_38Reshape%conv_res_block_1_34/Identity:output:0Reshape_38/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_38m
Identity_32IdentityMatMul_32:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_32
MatMul_33/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_33/ReadVariableOp
	MatMul_33MatMulIdentity_32:output:0 MatMul_33/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_33
(inner_decoder_34/StatefulPartitionedCallStatefulPartitionedCallMatMul_33:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_34/StatefulPartitionedCallÓ
inner_decoder_34/IdentityIdentity1inner_decoder_34/StatefulPartitionedCall:output:0)^inner_decoder_34/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_34/IdentityÙ
+conv_res_block_1_35/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_34/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_35/StatefulPartitionedCallß
conv_res_block_1_35/IdentityIdentity4conv_res_block_1_35/StatefulPartitionedCall:output:0,^conv_res_block_1_35/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_35/Identityy
Reshape_39/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_39/shape

Reshape_39Reshape%conv_res_block_1_35/Identity:output:0Reshape_39/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_39m
Identity_33IdentityMatMul_33:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_33
MatMul_34/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_34/ReadVariableOp
	MatMul_34MatMulIdentity_33:output:0 MatMul_34/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_34
(inner_decoder_35/StatefulPartitionedCallStatefulPartitionedCallMatMul_34:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_35/StatefulPartitionedCallÓ
inner_decoder_35/IdentityIdentity1inner_decoder_35/StatefulPartitionedCall:output:0)^inner_decoder_35/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_35/IdentityÙ
+conv_res_block_1_36/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_35/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_36/StatefulPartitionedCallß
conv_res_block_1_36/IdentityIdentity4conv_res_block_1_36/StatefulPartitionedCall:output:0,^conv_res_block_1_36/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_36/Identityy
Reshape_40/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_40/shape

Reshape_40Reshape%conv_res_block_1_36/Identity:output:0Reshape_40/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_40m
Identity_34IdentityMatMul_34:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_34
MatMul_35/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_35/ReadVariableOp
	MatMul_35MatMulIdentity_34:output:0 MatMul_35/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_35
(inner_decoder_36/StatefulPartitionedCallStatefulPartitionedCallMatMul_35:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_36/StatefulPartitionedCallÓ
inner_decoder_36/IdentityIdentity1inner_decoder_36/StatefulPartitionedCall:output:0)^inner_decoder_36/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_36/IdentityÙ
+conv_res_block_1_37/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_36/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_37/StatefulPartitionedCallß
conv_res_block_1_37/IdentityIdentity4conv_res_block_1_37/StatefulPartitionedCall:output:0,^conv_res_block_1_37/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_37/Identityy
Reshape_41/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_41/shape

Reshape_41Reshape%conv_res_block_1_37/Identity:output:0Reshape_41/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_41m
Identity_35IdentityMatMul_35:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_35
MatMul_36/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_36/ReadVariableOp
	MatMul_36MatMulIdentity_35:output:0 MatMul_36/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_36
(inner_decoder_37/StatefulPartitionedCallStatefulPartitionedCallMatMul_36:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_37/StatefulPartitionedCallÓ
inner_decoder_37/IdentityIdentity1inner_decoder_37/StatefulPartitionedCall:output:0)^inner_decoder_37/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_37/IdentityÙ
+conv_res_block_1_38/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_37/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_38/StatefulPartitionedCallß
conv_res_block_1_38/IdentityIdentity4conv_res_block_1_38/StatefulPartitionedCall:output:0,^conv_res_block_1_38/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_38/Identityy
Reshape_42/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_42/shape

Reshape_42Reshape%conv_res_block_1_38/Identity:output:0Reshape_42/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_42m
Identity_36IdentityMatMul_36:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_36
MatMul_37/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_37/ReadVariableOp
	MatMul_37MatMulIdentity_36:output:0 MatMul_37/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_37
(inner_decoder_38/StatefulPartitionedCallStatefulPartitionedCallMatMul_37:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_38/StatefulPartitionedCallÓ
inner_decoder_38/IdentityIdentity1inner_decoder_38/StatefulPartitionedCall:output:0)^inner_decoder_38/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_38/IdentityÙ
+conv_res_block_1_39/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_38/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_39/StatefulPartitionedCallß
conv_res_block_1_39/IdentityIdentity4conv_res_block_1_39/StatefulPartitionedCall:output:0,^conv_res_block_1_39/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_39/Identityy
Reshape_43/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_43/shape

Reshape_43Reshape%conv_res_block_1_39/Identity:output:0Reshape_43/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_43m
Identity_37IdentityMatMul_37:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_37
MatMul_38/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_38/ReadVariableOp
	MatMul_38MatMulIdentity_37:output:0 MatMul_38/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_38
(inner_decoder_39/StatefulPartitionedCallStatefulPartitionedCallMatMul_38:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_39/StatefulPartitionedCallÓ
inner_decoder_39/IdentityIdentity1inner_decoder_39/StatefulPartitionedCall:output:0)^inner_decoder_39/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_39/IdentityÙ
+conv_res_block_1_40/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_39/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_40/StatefulPartitionedCallß
conv_res_block_1_40/IdentityIdentity4conv_res_block_1_40/StatefulPartitionedCall:output:0,^conv_res_block_1_40/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_40/Identityy
Reshape_44/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_44/shape

Reshape_44Reshape%conv_res_block_1_40/Identity:output:0Reshape_44/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_44m
Identity_38IdentityMatMul_38:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_38
MatMul_39/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_39/ReadVariableOp
	MatMul_39MatMulIdentity_38:output:0 MatMul_39/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_39
(inner_decoder_40/StatefulPartitionedCallStatefulPartitionedCallMatMul_39:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_40/StatefulPartitionedCallÓ
inner_decoder_40/IdentityIdentity1inner_decoder_40/StatefulPartitionedCall:output:0)^inner_decoder_40/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_40/IdentityÙ
+conv_res_block_1_41/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_40/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_41/StatefulPartitionedCallß
conv_res_block_1_41/IdentityIdentity4conv_res_block_1_41/StatefulPartitionedCall:output:0,^conv_res_block_1_41/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_41/Identityy
Reshape_45/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_45/shape

Reshape_45Reshape%conv_res_block_1_41/Identity:output:0Reshape_45/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_45m
Identity_39IdentityMatMul_39:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_39
MatMul_40/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_40/ReadVariableOp
	MatMul_40MatMulIdentity_39:output:0 MatMul_40/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_40
(inner_decoder_41/StatefulPartitionedCallStatefulPartitionedCallMatMul_40:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_41/StatefulPartitionedCallÓ
inner_decoder_41/IdentityIdentity1inner_decoder_41/StatefulPartitionedCall:output:0)^inner_decoder_41/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_41/IdentityÙ
+conv_res_block_1_42/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_41/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_42/StatefulPartitionedCallß
conv_res_block_1_42/IdentityIdentity4conv_res_block_1_42/StatefulPartitionedCall:output:0,^conv_res_block_1_42/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_42/Identityy
Reshape_46/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_46/shape

Reshape_46Reshape%conv_res_block_1_42/Identity:output:0Reshape_46/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_46m
Identity_40IdentityMatMul_40:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_40
MatMul_41/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_41/ReadVariableOp
	MatMul_41MatMulIdentity_40:output:0 MatMul_41/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_41
(inner_decoder_42/StatefulPartitionedCallStatefulPartitionedCallMatMul_41:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_42/StatefulPartitionedCallÓ
inner_decoder_42/IdentityIdentity1inner_decoder_42/StatefulPartitionedCall:output:0)^inner_decoder_42/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_42/IdentityÙ
+conv_res_block_1_43/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_42/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_43/StatefulPartitionedCallß
conv_res_block_1_43/IdentityIdentity4conv_res_block_1_43/StatefulPartitionedCall:output:0,^conv_res_block_1_43/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_43/Identityy
Reshape_47/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_47/shape

Reshape_47Reshape%conv_res_block_1_43/Identity:output:0Reshape_47/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_47m
Identity_41IdentityMatMul_41:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_41
MatMul_42/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_42/ReadVariableOp
	MatMul_42MatMulIdentity_41:output:0 MatMul_42/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_42
(inner_decoder_43/StatefulPartitionedCallStatefulPartitionedCallMatMul_42:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_43/StatefulPartitionedCallÓ
inner_decoder_43/IdentityIdentity1inner_decoder_43/StatefulPartitionedCall:output:0)^inner_decoder_43/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_43/IdentityÙ
+conv_res_block_1_44/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_43/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_44/StatefulPartitionedCallß
conv_res_block_1_44/IdentityIdentity4conv_res_block_1_44/StatefulPartitionedCall:output:0,^conv_res_block_1_44/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_44/Identityy
Reshape_48/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_48/shape

Reshape_48Reshape%conv_res_block_1_44/Identity:output:0Reshape_48/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_48m
Identity_42IdentityMatMul_42:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_42
MatMul_43/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_43/ReadVariableOp
	MatMul_43MatMulIdentity_42:output:0 MatMul_43/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_43
(inner_decoder_44/StatefulPartitionedCallStatefulPartitionedCallMatMul_43:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_44/StatefulPartitionedCallÓ
inner_decoder_44/IdentityIdentity1inner_decoder_44/StatefulPartitionedCall:output:0)^inner_decoder_44/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_44/IdentityÙ
+conv_res_block_1_45/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_44/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_45/StatefulPartitionedCallß
conv_res_block_1_45/IdentityIdentity4conv_res_block_1_45/StatefulPartitionedCall:output:0,^conv_res_block_1_45/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_45/Identityy
Reshape_49/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_49/shape

Reshape_49Reshape%conv_res_block_1_45/Identity:output:0Reshape_49/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_49m
Identity_43IdentityMatMul_43:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_43
MatMul_44/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_44/ReadVariableOp
	MatMul_44MatMulIdentity_43:output:0 MatMul_44/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_44
(inner_decoder_45/StatefulPartitionedCallStatefulPartitionedCallMatMul_44:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_45/StatefulPartitionedCallÓ
inner_decoder_45/IdentityIdentity1inner_decoder_45/StatefulPartitionedCall:output:0)^inner_decoder_45/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_45/IdentityÙ
+conv_res_block_1_46/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_45/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_46/StatefulPartitionedCallß
conv_res_block_1_46/IdentityIdentity4conv_res_block_1_46/StatefulPartitionedCall:output:0,^conv_res_block_1_46/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_46/Identityy
Reshape_50/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_50/shape

Reshape_50Reshape%conv_res_block_1_46/Identity:output:0Reshape_50/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_50m
Identity_44IdentityMatMul_44:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_44
MatMul_45/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_45/ReadVariableOp
	MatMul_45MatMulIdentity_44:output:0 MatMul_45/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_45
(inner_decoder_46/StatefulPartitionedCallStatefulPartitionedCallMatMul_45:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_46/StatefulPartitionedCallÓ
inner_decoder_46/IdentityIdentity1inner_decoder_46/StatefulPartitionedCall:output:0)^inner_decoder_46/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_46/IdentityÙ
+conv_res_block_1_47/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_46/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_47/StatefulPartitionedCallß
conv_res_block_1_47/IdentityIdentity4conv_res_block_1_47/StatefulPartitionedCall:output:0,^conv_res_block_1_47/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_47/Identityy
Reshape_51/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_51/shape

Reshape_51Reshape%conv_res_block_1_47/Identity:output:0Reshape_51/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_51m
Identity_45IdentityMatMul_45:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_45
MatMul_46/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_46/ReadVariableOp
	MatMul_46MatMulIdentity_45:output:0 MatMul_46/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_46
(inner_decoder_47/StatefulPartitionedCallStatefulPartitionedCallMatMul_46:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_47/StatefulPartitionedCallÓ
inner_decoder_47/IdentityIdentity1inner_decoder_47/StatefulPartitionedCall:output:0)^inner_decoder_47/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_47/IdentityÙ
+conv_res_block_1_48/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_47/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_48/StatefulPartitionedCallß
conv_res_block_1_48/IdentityIdentity4conv_res_block_1_48/StatefulPartitionedCall:output:0,^conv_res_block_1_48/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_48/Identityy
Reshape_52/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_52/shape

Reshape_52Reshape%conv_res_block_1_48/Identity:output:0Reshape_52/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_52m
Identity_46IdentityMatMul_46:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_46
MatMul_47/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_47/ReadVariableOp
	MatMul_47MatMulIdentity_46:output:0 MatMul_47/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_47
(inner_decoder_48/StatefulPartitionedCallStatefulPartitionedCallMatMul_47:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_48/StatefulPartitionedCallÓ
inner_decoder_48/IdentityIdentity1inner_decoder_48/StatefulPartitionedCall:output:0)^inner_decoder_48/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_48/IdentityÙ
+conv_res_block_1_49/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_48/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_49/StatefulPartitionedCallß
conv_res_block_1_49/IdentityIdentity4conv_res_block_1_49/StatefulPartitionedCall:output:0,^conv_res_block_1_49/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_49/Identityy
Reshape_53/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_53/shape

Reshape_53Reshape%conv_res_block_1_49/Identity:output:0Reshape_53/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_53m
Identity_47IdentityMatMul_47:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_47
MatMul_48/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_48/ReadVariableOp
	MatMul_48MatMulIdentity_47:output:0 MatMul_48/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_48
(inner_decoder_49/StatefulPartitionedCallStatefulPartitionedCallMatMul_48:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_49/StatefulPartitionedCallÓ
inner_decoder_49/IdentityIdentity1inner_decoder_49/StatefulPartitionedCall:output:0)^inner_decoder_49/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_49/IdentityÙ
+conv_res_block_1_50/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_49/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_50/StatefulPartitionedCallß
conv_res_block_1_50/IdentityIdentity4conv_res_block_1_50/StatefulPartitionedCall:output:0,^conv_res_block_1_50/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_50/Identityy
Reshape_54/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_54/shape

Reshape_54Reshape%conv_res_block_1_50/Identity:output:0Reshape_54/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_54m
Identity_48IdentityMatMul_48:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_48
MatMul_49/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_49/ReadVariableOp
	MatMul_49MatMulIdentity_48:output:0 MatMul_49/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_49
(inner_decoder_50/StatefulPartitionedCallStatefulPartitionedCallMatMul_49:product:0inner_decoder_3112598*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_134693772*
(inner_decoder_50/StatefulPartitionedCallÓ
inner_decoder_50/IdentityIdentity1inner_decoder_50/StatefulPartitionedCall:output:0)^inner_decoder_50/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_decoder_50/IdentityÙ
+conv_res_block_1_51/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_50/Identity:output:0conv_res_block_1_3112773conv_res_block_1_3112775conv_res_block_1_3112777conv_res_block_1_3112779conv_res_block_1_3112781conv_res_block_1_3112783conv_res_block_1_3112785conv_res_block_1_3112787conv_res_block_1_3112789conv_res_block_1_3112791conv_res_block_1_3112793conv_res_block_1_3112795*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372-
+conv_res_block_1_51/StatefulPartitionedCallß
conv_res_block_1_51/IdentityIdentity4conv_res_block_1_51/StatefulPartitionedCall:output:0,^conv_res_block_1_51/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_1_51/Identityy
Reshape_55/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_55/shape

Reshape_55Reshape%conv_res_block_1_51/Identity:output:0Reshape_55/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_55m
Identity_49IdentityMatMul_49:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_49`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis	
concat_2ConcatV2Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0Reshape_16:output:0Reshape_17:output:0Reshape_18:output:0Reshape_19:output:0Reshape_20:output:0Reshape_21:output:0Reshape_22:output:0Reshape_23:output:0Reshape_24:output:0Reshape_25:output:0Reshape_26:output:0Reshape_27:output:0Reshape_28:output:0Reshape_29:output:0Reshape_30:output:0Reshape_31:output:0Reshape_32:output:0Reshape_33:output:0Reshape_34:output:0Reshape_35:output:0Reshape_36:output:0Reshape_37:output:0Reshape_38:output:0Reshape_39:output:0Reshape_40:output:0Reshape_41:output:0Reshape_42:output:0Reshape_43:output:0Reshape_44:output:0Reshape_45:output:0Reshape_46:output:0Reshape_47:output:0Reshape_48:output:0Reshape_49:output:0Reshape_50:output:0Reshape_51:output:0Reshape_52:output:0Reshape_53:output:0Reshape_54:output:0Reshape_55:output:0concat_2/axis:output:0*
N2*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

concat_2©
(conv_res_block_3/StatefulPartitionedCallStatefulPartitionedCallReshape_2:output:0conv_res_block_3112520conv_res_block_3112522conv_res_block_3112524conv_res_block_3112526conv_res_block_3112528conv_res_block_3112530conv_res_block_3112532conv_res_block_3112534conv_res_block_3112536conv_res_block_3112538conv_res_block_3112540conv_res_block_3112542*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*U
fPRN
L__inference_conv_res_block_layer_call_and_return_conditional_losses_134691662*
(conv_res_block_3/StatefulPartitionedCallÓ
conv_res_block_3/IdentityIdentity1conv_res_block_3/StatefulPartitionedCall:output:0)^conv_res_block_3/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_3/Identity
'inner_encoder_2/StatefulPartitionedCallStatefulPartitionedCall"conv_res_block_3/Identity:output:0inner_encoder_3112570*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_encoder_layer_call_and_return_conditional_losses_134702002)
'inner_encoder_2/StatefulPartitionedCallÎ
inner_encoder_2/IdentityIdentity0inner_encoder_2/StatefulPartitionedCall:output:0(^inner_encoder_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_encoder_2/Identity
MatMul_50/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_50/ReadVariableOp
	MatMul_50MatMul!inner_encoder_2/Identity:output:0 MatMul_50/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_50m
Identity_50IdentityMatMul_50:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_50y
Reshape_56/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_56/shape

Reshape_56ReshapeIdentity_50:output:0Reshape_56/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_56
MatMul_51/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_51/ReadVariableOp
	MatMul_51MatMulIdentity_50:output:0 MatMul_51/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_51m
Identity_51IdentityMatMul_51:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_51y
Reshape_57/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_57/shape

Reshape_57ReshapeIdentity_51:output:0Reshape_57/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_57
MatMul_52/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_52/ReadVariableOp
	MatMul_52MatMulIdentity_51:output:0 MatMul_52/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_52m
Identity_52IdentityMatMul_52:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_52y
Reshape_58/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_58/shape

Reshape_58ReshapeIdentity_52:output:0Reshape_58/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_58
MatMul_53/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_53/ReadVariableOp
	MatMul_53MatMulIdentity_52:output:0 MatMul_53/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_53m
Identity_53IdentityMatMul_53:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_53y
Reshape_59/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_59/shape

Reshape_59ReshapeIdentity_53:output:0Reshape_59/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_59
MatMul_54/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_54/ReadVariableOp
	MatMul_54MatMulIdentity_53:output:0 MatMul_54/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_54m
Identity_54IdentityMatMul_54:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_54y
Reshape_60/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_60/shape

Reshape_60ReshapeIdentity_54:output:0Reshape_60/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_60
MatMul_55/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_55/ReadVariableOp
	MatMul_55MatMulIdentity_54:output:0 MatMul_55/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_55m
Identity_55IdentityMatMul_55:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_55y
Reshape_61/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_61/shape

Reshape_61ReshapeIdentity_55:output:0Reshape_61/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_61
MatMul_56/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_56/ReadVariableOp
	MatMul_56MatMulIdentity_55:output:0 MatMul_56/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_56m
Identity_56IdentityMatMul_56:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_56y
Reshape_62/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_62/shape

Reshape_62ReshapeIdentity_56:output:0Reshape_62/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_62
MatMul_57/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_57/ReadVariableOp
	MatMul_57MatMulIdentity_56:output:0 MatMul_57/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_57m
Identity_57IdentityMatMul_57:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_57y
Reshape_63/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_63/shape

Reshape_63ReshapeIdentity_57:output:0Reshape_63/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_63
MatMul_58/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_58/ReadVariableOp
	MatMul_58MatMulIdentity_57:output:0 MatMul_58/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_58m
Identity_58IdentityMatMul_58:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_58y
Reshape_64/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_64/shape

Reshape_64ReshapeIdentity_58:output:0Reshape_64/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_64
MatMul_59/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_59/ReadVariableOp
	MatMul_59MatMulIdentity_58:output:0 MatMul_59/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_59m
Identity_59IdentityMatMul_59:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_59y
Reshape_65/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_65/shape

Reshape_65ReshapeIdentity_59:output:0Reshape_65/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_65
MatMul_60/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_60/ReadVariableOp
	MatMul_60MatMulIdentity_59:output:0 MatMul_60/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_60m
Identity_60IdentityMatMul_60:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_60y
Reshape_66/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_66/shape

Reshape_66ReshapeIdentity_60:output:0Reshape_66/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_66
MatMul_61/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_61/ReadVariableOp
	MatMul_61MatMulIdentity_60:output:0 MatMul_61/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_61m
Identity_61IdentityMatMul_61:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_61y
Reshape_67/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_67/shape

Reshape_67ReshapeIdentity_61:output:0Reshape_67/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_67
MatMul_62/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_62/ReadVariableOp
	MatMul_62MatMulIdentity_61:output:0 MatMul_62/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_62m
Identity_62IdentityMatMul_62:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_62y
Reshape_68/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_68/shape

Reshape_68ReshapeIdentity_62:output:0Reshape_68/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_68
MatMul_63/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_63/ReadVariableOp
	MatMul_63MatMulIdentity_62:output:0 MatMul_63/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_63m
Identity_63IdentityMatMul_63:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_63y
Reshape_69/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_69/shape

Reshape_69ReshapeIdentity_63:output:0Reshape_69/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_69
MatMul_64/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_64/ReadVariableOp
	MatMul_64MatMulIdentity_63:output:0 MatMul_64/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_64m
Identity_64IdentityMatMul_64:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_64y
Reshape_70/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_70/shape

Reshape_70ReshapeIdentity_64:output:0Reshape_70/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_70
MatMul_65/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_65/ReadVariableOp
	MatMul_65MatMulIdentity_64:output:0 MatMul_65/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_65m
Identity_65IdentityMatMul_65:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_65y
Reshape_71/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_71/shape

Reshape_71ReshapeIdentity_65:output:0Reshape_71/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_71
MatMul_66/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_66/ReadVariableOp
	MatMul_66MatMulIdentity_65:output:0 MatMul_66/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_66m
Identity_66IdentityMatMul_66:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_66y
Reshape_72/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_72/shape

Reshape_72ReshapeIdentity_66:output:0Reshape_72/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_72
MatMul_67/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_67/ReadVariableOp
	MatMul_67MatMulIdentity_66:output:0 MatMul_67/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_67m
Identity_67IdentityMatMul_67:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_67y
Reshape_73/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_73/shape

Reshape_73ReshapeIdentity_67:output:0Reshape_73/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_73
MatMul_68/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_68/ReadVariableOp
	MatMul_68MatMulIdentity_67:output:0 MatMul_68/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_68m
Identity_68IdentityMatMul_68:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_68y
Reshape_74/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_74/shape

Reshape_74ReshapeIdentity_68:output:0Reshape_74/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_74
MatMul_69/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_69/ReadVariableOp
	MatMul_69MatMulIdentity_68:output:0 MatMul_69/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_69m
Identity_69IdentityMatMul_69:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_69y
Reshape_75/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_75/shape

Reshape_75ReshapeIdentity_69:output:0Reshape_75/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_75
MatMul_70/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_70/ReadVariableOp
	MatMul_70MatMulIdentity_69:output:0 MatMul_70/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_70m
Identity_70IdentityMatMul_70:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_70y
Reshape_76/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_76/shape

Reshape_76ReshapeIdentity_70:output:0Reshape_76/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_76
MatMul_71/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_71/ReadVariableOp
	MatMul_71MatMulIdentity_70:output:0 MatMul_71/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_71m
Identity_71IdentityMatMul_71:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_71y
Reshape_77/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_77/shape

Reshape_77ReshapeIdentity_71:output:0Reshape_77/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_77
MatMul_72/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_72/ReadVariableOp
	MatMul_72MatMulIdentity_71:output:0 MatMul_72/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_72m
Identity_72IdentityMatMul_72:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_72y
Reshape_78/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_78/shape

Reshape_78ReshapeIdentity_72:output:0Reshape_78/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_78
MatMul_73/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_73/ReadVariableOp
	MatMul_73MatMulIdentity_72:output:0 MatMul_73/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_73m
Identity_73IdentityMatMul_73:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_73y
Reshape_79/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_79/shape

Reshape_79ReshapeIdentity_73:output:0Reshape_79/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_79
MatMul_74/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_74/ReadVariableOp
	MatMul_74MatMulIdentity_73:output:0 MatMul_74/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_74m
Identity_74IdentityMatMul_74:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_74y
Reshape_80/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_80/shape

Reshape_80ReshapeIdentity_74:output:0Reshape_80/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_80
MatMul_75/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_75/ReadVariableOp
	MatMul_75MatMulIdentity_74:output:0 MatMul_75/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_75m
Identity_75IdentityMatMul_75:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_75y
Reshape_81/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_81/shape

Reshape_81ReshapeIdentity_75:output:0Reshape_81/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_81
MatMul_76/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_76/ReadVariableOp
	MatMul_76MatMulIdentity_75:output:0 MatMul_76/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_76m
Identity_76IdentityMatMul_76:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_76y
Reshape_82/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_82/shape

Reshape_82ReshapeIdentity_76:output:0Reshape_82/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_82
MatMul_77/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_77/ReadVariableOp
	MatMul_77MatMulIdentity_76:output:0 MatMul_77/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_77m
Identity_77IdentityMatMul_77:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_77y
Reshape_83/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_83/shape

Reshape_83ReshapeIdentity_77:output:0Reshape_83/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_83
MatMul_78/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_78/ReadVariableOp
	MatMul_78MatMulIdentity_77:output:0 MatMul_78/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_78m
Identity_78IdentityMatMul_78:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_78y
Reshape_84/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_84/shape

Reshape_84ReshapeIdentity_78:output:0Reshape_84/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_84
MatMul_79/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_79/ReadVariableOp
	MatMul_79MatMulIdentity_78:output:0 MatMul_79/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_79m
Identity_79IdentityMatMul_79:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_79y
Reshape_85/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_85/shape

Reshape_85ReshapeIdentity_79:output:0Reshape_85/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_85
MatMul_80/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_80/ReadVariableOp
	MatMul_80MatMulIdentity_79:output:0 MatMul_80/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_80m
Identity_80IdentityMatMul_80:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_80y
Reshape_86/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_86/shape

Reshape_86ReshapeIdentity_80:output:0Reshape_86/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_86
MatMul_81/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_81/ReadVariableOp
	MatMul_81MatMulIdentity_80:output:0 MatMul_81/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_81m
Identity_81IdentityMatMul_81:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_81y
Reshape_87/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_87/shape

Reshape_87ReshapeIdentity_81:output:0Reshape_87/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_87
MatMul_82/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_82/ReadVariableOp
	MatMul_82MatMulIdentity_81:output:0 MatMul_82/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_82m
Identity_82IdentityMatMul_82:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_82y
Reshape_88/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_88/shape

Reshape_88ReshapeIdentity_82:output:0Reshape_88/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_88
MatMul_83/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_83/ReadVariableOp
	MatMul_83MatMulIdentity_82:output:0 MatMul_83/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_83m
Identity_83IdentityMatMul_83:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_83y
Reshape_89/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_89/shape

Reshape_89ReshapeIdentity_83:output:0Reshape_89/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_89
MatMul_84/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_84/ReadVariableOp
	MatMul_84MatMulIdentity_83:output:0 MatMul_84/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_84m
Identity_84IdentityMatMul_84:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_84y
Reshape_90/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_90/shape

Reshape_90ReshapeIdentity_84:output:0Reshape_90/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_90
MatMul_85/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_85/ReadVariableOp
	MatMul_85MatMulIdentity_84:output:0 MatMul_85/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_85m
Identity_85IdentityMatMul_85:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_85y
Reshape_91/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_91/shape

Reshape_91ReshapeIdentity_85:output:0Reshape_91/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_91
MatMul_86/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_86/ReadVariableOp
	MatMul_86MatMulIdentity_85:output:0 MatMul_86/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_86m
Identity_86IdentityMatMul_86:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_86y
Reshape_92/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_92/shape

Reshape_92ReshapeIdentity_86:output:0Reshape_92/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_92
MatMul_87/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_87/ReadVariableOp
	MatMul_87MatMulIdentity_86:output:0 MatMul_87/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_87m
Identity_87IdentityMatMul_87:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_87y
Reshape_93/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_93/shape

Reshape_93ReshapeIdentity_87:output:0Reshape_93/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_93
MatMul_88/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_88/ReadVariableOp
	MatMul_88MatMulIdentity_87:output:0 MatMul_88/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_88m
Identity_88IdentityMatMul_88:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_88y
Reshape_94/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_94/shape

Reshape_94ReshapeIdentity_88:output:0Reshape_94/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_94
MatMul_89/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_89/ReadVariableOp
	MatMul_89MatMulIdentity_88:output:0 MatMul_89/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_89m
Identity_89IdentityMatMul_89:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_89y
Reshape_95/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_95/shape

Reshape_95ReshapeIdentity_89:output:0Reshape_95/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_95
MatMul_90/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_90/ReadVariableOp
	MatMul_90MatMulIdentity_89:output:0 MatMul_90/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_90m
Identity_90IdentityMatMul_90:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_90y
Reshape_96/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_96/shape

Reshape_96ReshapeIdentity_90:output:0Reshape_96/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_96
MatMul_91/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_91/ReadVariableOp
	MatMul_91MatMulIdentity_90:output:0 MatMul_91/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_91m
Identity_91IdentityMatMul_91:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_91y
Reshape_97/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_97/shape

Reshape_97ReshapeIdentity_91:output:0Reshape_97/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_97
MatMul_92/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_92/ReadVariableOp
	MatMul_92MatMulIdentity_91:output:0 MatMul_92/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_92m
Identity_92IdentityMatMul_92:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_92y
Reshape_98/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_98/shape

Reshape_98ReshapeIdentity_92:output:0Reshape_98/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_98
MatMul_93/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_93/ReadVariableOp
	MatMul_93MatMulIdentity_92:output:0 MatMul_93/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_93m
Identity_93IdentityMatMul_93:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_93y
Reshape_99/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_99/shape

Reshape_99ReshapeIdentity_93:output:0Reshape_99/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Reshape_99
MatMul_94/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_94/ReadVariableOp
	MatMul_94MatMulIdentity_93:output:0 MatMul_94/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_94m
Identity_94IdentityMatMul_94:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_94{
Reshape_100/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_100/shape
Reshape_100ReshapeIdentity_94:output:0Reshape_100/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reshape_100
MatMul_95/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_95/ReadVariableOp
	MatMul_95MatMulIdentity_94:output:0 MatMul_95/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_95m
Identity_95IdentityMatMul_95:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_95{
Reshape_101/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_101/shape
Reshape_101ReshapeIdentity_95:output:0Reshape_101/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reshape_101
MatMul_96/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_96/ReadVariableOp
	MatMul_96MatMulIdentity_95:output:0 MatMul_96/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_96m
Identity_96IdentityMatMul_96:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_96{
Reshape_102/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_102/shape
Reshape_102ReshapeIdentity_96:output:0Reshape_102/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reshape_102
MatMul_97/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_97/ReadVariableOp
	MatMul_97MatMulIdentity_96:output:0 MatMul_97/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_97m
Identity_97IdentityMatMul_97:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_97{
Reshape_103/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_103/shape
Reshape_103ReshapeIdentity_97:output:0Reshape_103/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reshape_103
MatMul_98/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_98/ReadVariableOp
	MatMul_98MatMulIdentity_97:output:0 MatMul_98/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_98m
Identity_98IdentityMatMul_98:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_98{
Reshape_104/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_104/shape
Reshape_104ReshapeIdentity_98:output:0Reshape_104/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reshape_104
MatMul_99/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_99/ReadVariableOp
	MatMul_99MatMulIdentity_98:output:0 MatMul_99/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MatMul_99m
Identity_99IdentityMatMul_99:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_99{
Reshape_105/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2
Reshape_105/shape
Reshape_105ReshapeIdentity_99:output:0Reshape_105/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reshape_105`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_3/axis	
concat_3ConcatV2Reshape_56:output:0Reshape_57:output:0Reshape_58:output:0Reshape_59:output:0Reshape_60:output:0Reshape_61:output:0Reshape_62:output:0Reshape_63:output:0Reshape_64:output:0Reshape_65:output:0Reshape_66:output:0Reshape_67:output:0Reshape_68:output:0Reshape_69:output:0Reshape_70:output:0Reshape_71:output:0Reshape_72:output:0Reshape_73:output:0Reshape_74:output:0Reshape_75:output:0Reshape_76:output:0Reshape_77:output:0Reshape_78:output:0Reshape_79:output:0Reshape_80:output:0Reshape_81:output:0Reshape_82:output:0Reshape_83:output:0Reshape_84:output:0Reshape_85:output:0Reshape_86:output:0Reshape_87:output:0Reshape_88:output:0Reshape_89:output:0Reshape_90:output:0Reshape_91:output:0Reshape_92:output:0Reshape_93:output:0Reshape_94:output:0Reshape_95:output:0Reshape_96:output:0Reshape_97:output:0Reshape_98:output:0Reshape_99:output:0Reshape_100:output:0Reshape_101:output:0Reshape_102:output:0Reshape_103:output:0Reshape_104:output:0Reshape_105:output:0concat_3/axis:output:0*
N2*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

concat_3©
(conv_res_block_4/StatefulPartitionedCallStatefulPartitionedCallReshape_3:output:0conv_res_block_3112520conv_res_block_3112522conv_res_block_3112524conv_res_block_3112526conv_res_block_3112528conv_res_block_3112530conv_res_block_3112532conv_res_block_3112534conv_res_block_3112536conv_res_block_3112538conv_res_block_3112540conv_res_block_3112542*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*U
fPRN
L__inference_conv_res_block_layer_call_and_return_conditional_losses_134691662*
(conv_res_block_4/StatefulPartitionedCallÓ
conv_res_block_4/IdentityIdentity1conv_res_block_4/StatefulPartitionedCall:output:0)^conv_res_block_4/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_res_block_4/Identity
'inner_encoder_3/StatefulPartitionedCallStatefulPartitionedCall"conv_res_block_4/Identity:output:0inner_encoder_3112570*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_encoder_layer_call_and_return_conditional_losses_134702002)
'inner_encoder_3/StatefulPartitionedCallÎ
inner_encoder_3/IdentityIdentity0inner_encoder_3/StatefulPartitionedCall:output:0(^inner_encoder_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
inner_encoder_3/Identity{
Reshape_106/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ2      2
Reshape_106/shape
Reshape_106Reshape!inner_encoder_3/Identity:output:0Reshape_106/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Reshape_106
RelMSE_1/subSubconcat_3:output:0Reshape_106:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
RelMSE_1/subt
RelMSE_1/SquareSquareRelMSE_1/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
RelMSE_1/Square
RelMSE_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
RelMSE_1/Mean/reduction_indices
RelMSE_1/MeanMeanRelMSE_1/Square:y:0(RelMSE_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
RelMSE_1/Mean|
RelMSE_1/Square_1SquareReshape_106:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
RelMSE_1/Square_1
!RelMSE_1/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!RelMSE_1/Mean_1/reduction_indices
RelMSE_1/Mean_1MeanRelMSE_1/Square_1:y:0*RelMSE_1/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
RelMSE_1/Mean_1e
RelMSE_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'72
RelMSE_1/add/y
RelMSE_1/addAddV2RelMSE_1/Mean_1:output:0RelMSE_1/add/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
RelMSE_1/add
RelMSE_1/truedivRealDivRelMSE_1/Mean:output:0RelMSE_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
RelMSE_1/truediv
!RelMSE_1/Mean_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!RelMSE_1/Mean_2/reduction_indices
RelMSE_1/Mean_2MeanRelMSE_1/truediv:z:0*RelMSE_1/Mean_2/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
RelMSE_1/Mean_2
RelMSE_1/weighted_loss/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
RelMSE_1/weighted_loss/Cast/x¯
RelMSE_1/weighted_loss/MulMulRelMSE_1/Mean_2:output:0&RelMSE_1/weighted_loss/Cast/x:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
RelMSE_1/weighted_loss/Mul
RelMSE_1/weighted_loss/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
RelMSE_1/weighted_loss/Const§
RelMSE_1/weighted_loss/SumSumRelMSE_1/weighted_loss/Mul:z:0%RelMSE_1/weighted_loss/Const:output:0*
T0*
_output_shapes
: 2
RelMSE_1/weighted_loss/Sum
#RelMSE_1/weighted_loss/num_elementsSizeRelMSE_1/weighted_loss/Mul:z:0*
T0*
_output_shapes
: 2%
#RelMSE_1/weighted_loss/num_elementsº
(RelMSE_1/weighted_loss/num_elements/CastCast,RelMSE_1/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(RelMSE_1/weighted_loss/num_elements/Cast
RelMSE_1/weighted_loss/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2 
RelMSE_1/weighted_loss/Const_1²
RelMSE_1/weighted_loss/Sum_1Sum#RelMSE_1/weighted_loss/Sum:output:0'RelMSE_1/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 2
RelMSE_1/weighted_loss/Sum_1¾
RelMSE_1/weighted_loss/valueDivNoNan%RelMSE_1/weighted_loss/Sum_1:output:0,RelMSE_1/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2
RelMSE_1/weighted_loss/valueW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_1/xj
mul_1Mulmul_1/x:output:0 RelMSE_1/weighted_loss/value:z:0*
T0*
_output_shapes
: 2
mul_1õ
Knetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_res_block_3112520*"
_output_shapes
:*
dtype02M
Knetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Square/ReadVariableOp
<network_arch/conv_res_block/conv1d/kernel/Regularizer/SquareSquareSnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2>
<network_arch/conv_res_block/conv1d/kernel/Regularizer/SquareÏ
;network_arch/conv_res_block/conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2=
;network_arch/conv_res_block/conv1d/kernel/Regularizer/Const¦
9network_arch/conv_res_block/conv1d/kernel/Regularizer/SumSum@network_arch/conv_res_block/conv1d/kernel/Regularizer/Square:y:0Dnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2;
9network_arch/conv_res_block/conv1d/kernel/Regularizer/Sum¿
;network_arch/conv_res_block/conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22=
;network_arch/conv_res_block/conv1d/kernel/Regularizer/mul/x¨
9network_arch/conv_res_block/conv1d/kernel/Regularizer/mulMulDnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/mul/x:output:0Bnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2;
9network_arch/conv_res_block/conv1d/kernel/Regularizer/mul¿
;network_arch/conv_res_block/conv1d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2=
;network_arch/conv_res_block/conv1d/kernel/Regularizer/add/x¥
9network_arch/conv_res_block/conv1d/kernel/Regularizer/addAddV2Dnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/add/x:output:0=network_arch/conv_res_block/conv1d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2;
9network_arch/conv_res_block/conv1d/kernel/Regularizer/addù
Mnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_res_block_3112524*"
_output_shapes
:*
dtype02O
Mnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Square/ReadVariableOp
>network_arch/conv_res_block/conv1d_1/kernel/Regularizer/SquareSquareUnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2@
>network_arch/conv_res_block/conv1d_1/kernel/Regularizer/SquareÓ
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/Const®
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/SumSumBnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Square:y:0Fnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/SumÃ
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22?
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mul/x°
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mulMulFnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/mul/x:output:0Dnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mulÃ
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/add/x­
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/addAddV2Fnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/add/x:output:0?network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/addù
Mnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_res_block_3112528*"
_output_shapes
: *
dtype02O
Mnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Square/ReadVariableOp
>network_arch/conv_res_block/conv1d_2/kernel/Regularizer/SquareSquareUnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2@
>network_arch/conv_res_block/conv1d_2/kernel/Regularizer/SquareÓ
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/Const®
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/SumSumBnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Square:y:0Fnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/SumÃ
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22?
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mul/x°
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mulMulFnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/mul/x:output:0Dnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mulÃ
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/add/x­
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/addAddV2Fnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/add/x:output:0?network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/addù
Mnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_res_block_3112532*"
_output_shapes
: @*
dtype02O
Mnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Square/ReadVariableOp
>network_arch/conv_res_block/conv1d_3/kernel/Regularizer/SquareSquareUnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2@
>network_arch/conv_res_block/conv1d_3/kernel/Regularizer/SquareÓ
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/Const®
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/SumSumBnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Square:y:0Fnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/SumÃ
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22?
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mul/x°
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mulMulFnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/mul/x:output:0Dnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mulÃ
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/add/x­
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/addAddV2Fnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/add/x:output:0?network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/addñ
Jnetwork_arch/conv_res_block/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_res_block_3112536* 
_output_shapes
:
*
dtype02L
Jnetwork_arch/conv_res_block/dense/kernel/Regularizer/Square/ReadVariableOp
;network_arch/conv_res_block/dense/kernel/Regularizer/SquareSquareRnetwork_arch/conv_res_block/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2=
;network_arch/conv_res_block/dense/kernel/Regularizer/SquareÉ
:network_arch/conv_res_block/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2<
:network_arch/conv_res_block/dense/kernel/Regularizer/Const¢
8network_arch/conv_res_block/dense/kernel/Regularizer/SumSum?network_arch/conv_res_block/dense/kernel/Regularizer/Square:y:0Cnetwork_arch/conv_res_block/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2:
8network_arch/conv_res_block/dense/kernel/Regularizer/Sum½
:network_arch/conv_res_block/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22<
:network_arch/conv_res_block/dense/kernel/Regularizer/mul/x¤
8network_arch/conv_res_block/dense/kernel/Regularizer/mulMulCnetwork_arch/conv_res_block/dense/kernel/Regularizer/mul/x:output:0Anetwork_arch/conv_res_block/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2:
8network_arch/conv_res_block/dense/kernel/Regularizer/mul½
:network_arch/conv_res_block/dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2<
:network_arch/conv_res_block/dense/kernel/Regularizer/add/x¡
8network_arch/conv_res_block/dense/kernel/Regularizer/addAddV2Cnetwork_arch/conv_res_block/dense/kernel/Regularizer/add/x:output:0<network_arch/conv_res_block/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2:
8network_arch/conv_res_block/dense/kernel/Regularizer/addõ
Lnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_res_block_3112540* 
_output_shapes
:
*
dtype02N
Lnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Square/ReadVariableOp
=network_arch/conv_res_block/dense_1/kernel/Regularizer/SquareSquareTnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2?
=network_arch/conv_res_block/dense_1/kernel/Regularizer/SquareÍ
<network_arch/conv_res_block/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2>
<network_arch/conv_res_block/dense_1/kernel/Regularizer/Constª
:network_arch/conv_res_block/dense_1/kernel/Regularizer/SumSumAnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Square:y:0Enetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2<
:network_arch/conv_res_block/dense_1/kernel/Regularizer/SumÁ
<network_arch/conv_res_block/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22>
<network_arch/conv_res_block/dense_1/kernel/Regularizer/mul/x¬
:network_arch/conv_res_block/dense_1/kernel/Regularizer/mulMulEnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/mul/x:output:0Cnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2<
:network_arch/conv_res_block/dense_1/kernel/Regularizer/mulÁ
<network_arch/conv_res_block/dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2>
<network_arch/conv_res_block/dense_1/kernel/Regularizer/add/x©
:network_arch/conv_res_block/dense_1/kernel/Regularizer/addAddV2Enetwork_arch/conv_res_block/dense_1/kernel/Regularizer/add/x:output:0>network_arch/conv_res_block/dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2<
:network_arch/conv_res_block/dense_1/kernel/Regularizer/addÿ
Onetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_res_block_1_3112773*"
_output_shapes
:*
dtype02Q
Onetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square/ReadVariableOp
@network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/SquareSquareWnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2B
@network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square×
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Const¶
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/SumSumDnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Square:y:0Hnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/SumÇ
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22A
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mul/x¸
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mulMulHnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mul/x:output:0Fnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mulÇ
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/add/xµ
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/addAddV2Hnetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/add/x:output:0Anetwork_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_4/kernel/Regularizer/addÿ
Onetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_res_block_1_3112777*"
_output_shapes
:*
dtype02Q
Onetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square/ReadVariableOp
@network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/SquareSquareWnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2B
@network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square×
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Const¶
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/SumSumDnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Square:y:0Hnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/SumÇ
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22A
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mul/x¸
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mulMulHnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mul/x:output:0Fnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mulÇ
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/add/xµ
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/addAddV2Hnetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/add/x:output:0Anetwork_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_5/kernel/Regularizer/addÿ
Onetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_res_block_1_3112781*"
_output_shapes
: *
dtype02Q
Onetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square/ReadVariableOp
@network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/SquareSquareWnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2B
@network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square×
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Const¶
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/SumSumDnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Square:y:0Hnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/SumÇ
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22A
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mul/x¸
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mulMulHnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mul/x:output:0Fnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mulÇ
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/add/xµ
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/addAddV2Hnetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/add/x:output:0Anetwork_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_6/kernel/Regularizer/addÿ
Onetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_res_block_1_3112785*"
_output_shapes
: @*
dtype02Q
Onetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square/ReadVariableOp
@network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/SquareSquareWnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2B
@network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square×
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Const¶
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/SumSumDnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Square:y:0Hnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/SumÇ
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22A
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mul/x¸
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mulMulHnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mul/x:output:0Fnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mulÇ
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/add/xµ
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/addAddV2Hnetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/add/x:output:0Anetwork_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/conv_res_block_1/conv1d_7/kernel/Regularizer/addû
Nnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_res_block_1_3112789* 
_output_shapes
:
*
dtype02P
Nnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Square/ReadVariableOp
?network_arch/conv_res_block_1/dense_2/kernel/Regularizer/SquareSquareVnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2A
?network_arch/conv_res_block_1/dense_2/kernel/Regularizer/SquareÑ
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2@
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/Const²
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/SumSumCnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Square:y:0Gnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/SumÅ
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22@
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mul/x´
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mulMulGnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/mul/x:output:0Enetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mulÅ
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>network_arch/conv_res_block_1/dense_2/kernel/Regularizer/add/x±
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/addAddV2Gnetwork_arch/conv_res_block_1/dense_2/kernel/Regularizer/add/x:output:0@network_arch/conv_res_block_1/dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_2/kernel/Regularizer/addû
Nnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_res_block_1_3112793* 
_output_shapes
:
*
dtype02P
Nnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Square/ReadVariableOp
?network_arch/conv_res_block_1/dense_3/kernel/Regularizer/SquareSquareVnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2A
?network_arch/conv_res_block_1/dense_3/kernel/Regularizer/SquareÑ
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2@
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/Const²
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/SumSumCnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Square:y:0Gnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/SumÅ
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22@
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mul/x´
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mulMulGnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/mul/x:output:0Enetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mulÅ
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>network_arch/conv_res_block_1/dense_3/kernel/Regularizer/add/x±
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/addAddV2Gnetwork_arch/conv_res_block_1/dense_3/kernel/Regularizer/add/x:output:0@network_arch/conv_res_block_1/dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2>
<network_arch/conv_res_block_1/dense_3/kernel/Regularizer/addá
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpinner_encoder_3112570*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpí
4network_arch/inner_encoder/kernel/Regularizer/SquareSquareKnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	26
4network_arch/inner_encoder/kernel/Regularizer/Square»
3network_arch/inner_encoder/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       25
3network_arch/inner_encoder/kernel/Regularizer/Const
1network_arch/inner_encoder/kernel/Regularizer/SumSum8network_arch/inner_encoder/kernel/Regularizer/Square:y:0<network_arch/inner_encoder/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/Sum¯
3network_arch/inner_encoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+225
3network_arch/inner_encoder/kernel/Regularizer/mul/x
1network_arch/inner_encoder/kernel/Regularizer/mulMul<network_arch/inner_encoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_encoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/mul¯
3network_arch/inner_encoder/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3network_arch/inner_encoder/kernel/Regularizer/add/x
1network_arch/inner_encoder/kernel/Regularizer/addAddV2<network_arch/inner_encoder/kernel/Regularizer/add/x:output:05network_arch/inner_encoder/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/addá
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpinner_decoder_3112598*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpí
4network_arch/inner_decoder/kernel/Regularizer/SquareSquareKnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	26
4network_arch/inner_decoder/kernel/Regularizer/Square»
3network_arch/inner_decoder/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       25
3network_arch/inner_decoder/kernel/Regularizer/Const
1network_arch/inner_decoder/kernel/Regularizer/SumSum8network_arch/inner_decoder/kernel/Regularizer/Square:y:0<network_arch/inner_decoder/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/Sum¯
3network_arch/inner_decoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+225
3network_arch/inner_decoder/kernel/Regularizer/mul/x
1network_arch/inner_decoder/kernel/Regularizer/mulMul<network_arch/inner_decoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_decoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/mul¯
3network_arch/inner_decoder/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3network_arch/inner_decoder/kernel/Regularizer/add/x
1network_arch/inner_decoder/kernel/Regularizer/addAddV2<network_arch/inner_decoder/kernel/Regularizer/add/x:output:05network_arch/inner_decoder/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/add'
Identity_100IdentityReshape_4:output:0'^conv_res_block/StatefulPartitionedCall)^conv_res_block_1/StatefulPartitionedCall+^conv_res_block_1_1/StatefulPartitionedCall,^conv_res_block_1_10/StatefulPartitionedCall,^conv_res_block_1_11/StatefulPartitionedCall,^conv_res_block_1_12/StatefulPartitionedCall,^conv_res_block_1_13/StatefulPartitionedCall,^conv_res_block_1_14/StatefulPartitionedCall,^conv_res_block_1_15/StatefulPartitionedCall,^conv_res_block_1_16/StatefulPartitionedCall,^conv_res_block_1_17/StatefulPartitionedCall,^conv_res_block_1_18/StatefulPartitionedCall,^conv_res_block_1_19/StatefulPartitionedCall+^conv_res_block_1_2/StatefulPartitionedCall,^conv_res_block_1_20/StatefulPartitionedCall,^conv_res_block_1_21/StatefulPartitionedCall,^conv_res_block_1_22/StatefulPartitionedCall,^conv_res_block_1_23/StatefulPartitionedCall,^conv_res_block_1_24/StatefulPartitionedCall,^conv_res_block_1_25/StatefulPartitionedCall,^conv_res_block_1_26/StatefulPartitionedCall,^conv_res_block_1_27/StatefulPartitionedCall,^conv_res_block_1_28/StatefulPartitionedCall,^conv_res_block_1_29/StatefulPartitionedCall+^conv_res_block_1_3/StatefulPartitionedCall,^conv_res_block_1_30/StatefulPartitionedCall,^conv_res_block_1_31/StatefulPartitionedCall,^conv_res_block_1_32/StatefulPartitionedCall,^conv_res_block_1_33/StatefulPartitionedCall,^conv_res_block_1_34/StatefulPartitionedCall,^conv_res_block_1_35/StatefulPartitionedCall,^conv_res_block_1_36/StatefulPartitionedCall,^conv_res_block_1_37/StatefulPartitionedCall,^conv_res_block_1_38/StatefulPartitionedCall,^conv_res_block_1_39/StatefulPartitionedCall+^conv_res_block_1_4/StatefulPartitionedCall,^conv_res_block_1_40/StatefulPartitionedCall,^conv_res_block_1_41/StatefulPartitionedCall,^conv_res_block_1_42/StatefulPartitionedCall,^conv_res_block_1_43/StatefulPartitionedCall,^conv_res_block_1_44/StatefulPartitionedCall,^conv_res_block_1_45/StatefulPartitionedCall,^conv_res_block_1_46/StatefulPartitionedCall,^conv_res_block_1_47/StatefulPartitionedCall,^conv_res_block_1_48/StatefulPartitionedCall,^conv_res_block_1_49/StatefulPartitionedCall+^conv_res_block_1_5/StatefulPartitionedCall,^conv_res_block_1_50/StatefulPartitionedCall,^conv_res_block_1_51/StatefulPartitionedCall+^conv_res_block_1_6/StatefulPartitionedCall+^conv_res_block_1_7/StatefulPartitionedCall+^conv_res_block_1_8/StatefulPartitionedCall+^conv_res_block_1_9/StatefulPartitionedCall)^conv_res_block_2/StatefulPartitionedCall)^conv_res_block_3/StatefulPartitionedCall)^conv_res_block_4/StatefulPartitionedCall&^inner_decoder/StatefulPartitionedCall(^inner_decoder_1/StatefulPartitionedCall)^inner_decoder_10/StatefulPartitionedCall)^inner_decoder_11/StatefulPartitionedCall)^inner_decoder_12/StatefulPartitionedCall)^inner_decoder_13/StatefulPartitionedCall)^inner_decoder_14/StatefulPartitionedCall)^inner_decoder_15/StatefulPartitionedCall)^inner_decoder_16/StatefulPartitionedCall)^inner_decoder_17/StatefulPartitionedCall)^inner_decoder_18/StatefulPartitionedCall)^inner_decoder_19/StatefulPartitionedCall(^inner_decoder_2/StatefulPartitionedCall)^inner_decoder_20/StatefulPartitionedCall)^inner_decoder_21/StatefulPartitionedCall)^inner_decoder_22/StatefulPartitionedCall)^inner_decoder_23/StatefulPartitionedCall)^inner_decoder_24/StatefulPartitionedCall)^inner_decoder_25/StatefulPartitionedCall)^inner_decoder_26/StatefulPartitionedCall)^inner_decoder_27/StatefulPartitionedCall)^inner_decoder_28/StatefulPartitionedCall)^inner_decoder_29/StatefulPartitionedCall(^inner_decoder_3/StatefulPartitionedCall)^inner_decoder_30/StatefulPartitionedCall)^inner_decoder_31/StatefulPartitionedCall)^inner_decoder_32/StatefulPartitionedCall)^inner_decoder_33/StatefulPartitionedCall)^inner_decoder_34/StatefulPartitionedCall)^inner_decoder_35/StatefulPartitionedCall)^inner_decoder_36/StatefulPartitionedCall)^inner_decoder_37/StatefulPartitionedCall)^inner_decoder_38/StatefulPartitionedCall)^inner_decoder_39/StatefulPartitionedCall(^inner_decoder_4/StatefulPartitionedCall)^inner_decoder_40/StatefulPartitionedCall)^inner_decoder_41/StatefulPartitionedCall)^inner_decoder_42/StatefulPartitionedCall)^inner_decoder_43/StatefulPartitionedCall)^inner_decoder_44/StatefulPartitionedCall)^inner_decoder_45/StatefulPartitionedCall)^inner_decoder_46/StatefulPartitionedCall)^inner_decoder_47/StatefulPartitionedCall)^inner_decoder_48/StatefulPartitionedCall)^inner_decoder_49/StatefulPartitionedCall(^inner_decoder_5/StatefulPartitionedCall)^inner_decoder_50/StatefulPartitionedCall(^inner_decoder_6/StatefulPartitionedCall(^inner_decoder_7/StatefulPartitionedCall(^inner_decoder_8/StatefulPartitionedCall(^inner_decoder_9/StatefulPartitionedCall&^inner_encoder/StatefulPartitionedCall(^inner_encoder_1/StatefulPartitionedCall(^inner_encoder_2/StatefulPartitionedCall(^inner_encoder_3/StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
Identity_100'
Identity_101IdentityReshape_5:output:0'^conv_res_block/StatefulPartitionedCall)^conv_res_block_1/StatefulPartitionedCall+^conv_res_block_1_1/StatefulPartitionedCall,^conv_res_block_1_10/StatefulPartitionedCall,^conv_res_block_1_11/StatefulPartitionedCall,^conv_res_block_1_12/StatefulPartitionedCall,^conv_res_block_1_13/StatefulPartitionedCall,^conv_res_block_1_14/StatefulPartitionedCall,^conv_res_block_1_15/StatefulPartitionedCall,^conv_res_block_1_16/StatefulPartitionedCall,^conv_res_block_1_17/StatefulPartitionedCall,^conv_res_block_1_18/StatefulPartitionedCall,^conv_res_block_1_19/StatefulPartitionedCall+^conv_res_block_1_2/StatefulPartitionedCall,^conv_res_block_1_20/StatefulPartitionedCall,^conv_res_block_1_21/StatefulPartitionedCall,^conv_res_block_1_22/StatefulPartitionedCall,^conv_res_block_1_23/StatefulPartitionedCall,^conv_res_block_1_24/StatefulPartitionedCall,^conv_res_block_1_25/StatefulPartitionedCall,^conv_res_block_1_26/StatefulPartitionedCall,^conv_res_block_1_27/StatefulPartitionedCall,^conv_res_block_1_28/StatefulPartitionedCall,^conv_res_block_1_29/StatefulPartitionedCall+^conv_res_block_1_3/StatefulPartitionedCall,^conv_res_block_1_30/StatefulPartitionedCall,^conv_res_block_1_31/StatefulPartitionedCall,^conv_res_block_1_32/StatefulPartitionedCall,^conv_res_block_1_33/StatefulPartitionedCall,^conv_res_block_1_34/StatefulPartitionedCall,^conv_res_block_1_35/StatefulPartitionedCall,^conv_res_block_1_36/StatefulPartitionedCall,^conv_res_block_1_37/StatefulPartitionedCall,^conv_res_block_1_38/StatefulPartitionedCall,^conv_res_block_1_39/StatefulPartitionedCall+^conv_res_block_1_4/StatefulPartitionedCall,^conv_res_block_1_40/StatefulPartitionedCall,^conv_res_block_1_41/StatefulPartitionedCall,^conv_res_block_1_42/StatefulPartitionedCall,^conv_res_block_1_43/StatefulPartitionedCall,^conv_res_block_1_44/StatefulPartitionedCall,^conv_res_block_1_45/StatefulPartitionedCall,^conv_res_block_1_46/StatefulPartitionedCall,^conv_res_block_1_47/StatefulPartitionedCall,^conv_res_block_1_48/StatefulPartitionedCall,^conv_res_block_1_49/StatefulPartitionedCall+^conv_res_block_1_5/StatefulPartitionedCall,^conv_res_block_1_50/StatefulPartitionedCall,^conv_res_block_1_51/StatefulPartitionedCall+^conv_res_block_1_6/StatefulPartitionedCall+^conv_res_block_1_7/StatefulPartitionedCall+^conv_res_block_1_8/StatefulPartitionedCall+^conv_res_block_1_9/StatefulPartitionedCall)^conv_res_block_2/StatefulPartitionedCall)^conv_res_block_3/StatefulPartitionedCall)^conv_res_block_4/StatefulPartitionedCall&^inner_decoder/StatefulPartitionedCall(^inner_decoder_1/StatefulPartitionedCall)^inner_decoder_10/StatefulPartitionedCall)^inner_decoder_11/StatefulPartitionedCall)^inner_decoder_12/StatefulPartitionedCall)^inner_decoder_13/StatefulPartitionedCall)^inner_decoder_14/StatefulPartitionedCall)^inner_decoder_15/StatefulPartitionedCall)^inner_decoder_16/StatefulPartitionedCall)^inner_decoder_17/StatefulPartitionedCall)^inner_decoder_18/StatefulPartitionedCall)^inner_decoder_19/StatefulPartitionedCall(^inner_decoder_2/StatefulPartitionedCall)^inner_decoder_20/StatefulPartitionedCall)^inner_decoder_21/StatefulPartitionedCall)^inner_decoder_22/StatefulPartitionedCall)^inner_decoder_23/StatefulPartitionedCall)^inner_decoder_24/StatefulPartitionedCall)^inner_decoder_25/StatefulPartitionedCall)^inner_decoder_26/StatefulPartitionedCall)^inner_decoder_27/StatefulPartitionedCall)^inner_decoder_28/StatefulPartitionedCall)^inner_decoder_29/StatefulPartitionedCall(^inner_decoder_3/StatefulPartitionedCall)^inner_decoder_30/StatefulPartitionedCall)^inner_decoder_31/StatefulPartitionedCall)^inner_decoder_32/StatefulPartitionedCall)^inner_decoder_33/StatefulPartitionedCall)^inner_decoder_34/StatefulPartitionedCall)^inner_decoder_35/StatefulPartitionedCall)^inner_decoder_36/StatefulPartitionedCall)^inner_decoder_37/StatefulPartitionedCall)^inner_decoder_38/StatefulPartitionedCall)^inner_decoder_39/StatefulPartitionedCall(^inner_decoder_4/StatefulPartitionedCall)^inner_decoder_40/StatefulPartitionedCall)^inner_decoder_41/StatefulPartitionedCall)^inner_decoder_42/StatefulPartitionedCall)^inner_decoder_43/StatefulPartitionedCall)^inner_decoder_44/StatefulPartitionedCall)^inner_decoder_45/StatefulPartitionedCall)^inner_decoder_46/StatefulPartitionedCall)^inner_decoder_47/StatefulPartitionedCall)^inner_decoder_48/StatefulPartitionedCall)^inner_decoder_49/StatefulPartitionedCall(^inner_decoder_5/StatefulPartitionedCall)^inner_decoder_50/StatefulPartitionedCall(^inner_decoder_6/StatefulPartitionedCall(^inner_decoder_7/StatefulPartitionedCall(^inner_decoder_8/StatefulPartitionedCall(^inner_decoder_9/StatefulPartitionedCall&^inner_encoder/StatefulPartitionedCall(^inner_encoder_1/StatefulPartitionedCall(^inner_encoder_2/StatefulPartitionedCall(^inner_encoder_3/StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
Identity_101'
Identity_102Identityconcat_2:output:0'^conv_res_block/StatefulPartitionedCall)^conv_res_block_1/StatefulPartitionedCall+^conv_res_block_1_1/StatefulPartitionedCall,^conv_res_block_1_10/StatefulPartitionedCall,^conv_res_block_1_11/StatefulPartitionedCall,^conv_res_block_1_12/StatefulPartitionedCall,^conv_res_block_1_13/StatefulPartitionedCall,^conv_res_block_1_14/StatefulPartitionedCall,^conv_res_block_1_15/StatefulPartitionedCall,^conv_res_block_1_16/StatefulPartitionedCall,^conv_res_block_1_17/StatefulPartitionedCall,^conv_res_block_1_18/StatefulPartitionedCall,^conv_res_block_1_19/StatefulPartitionedCall+^conv_res_block_1_2/StatefulPartitionedCall,^conv_res_block_1_20/StatefulPartitionedCall,^conv_res_block_1_21/StatefulPartitionedCall,^conv_res_block_1_22/StatefulPartitionedCall,^conv_res_block_1_23/StatefulPartitionedCall,^conv_res_block_1_24/StatefulPartitionedCall,^conv_res_block_1_25/StatefulPartitionedCall,^conv_res_block_1_26/StatefulPartitionedCall,^conv_res_block_1_27/StatefulPartitionedCall,^conv_res_block_1_28/StatefulPartitionedCall,^conv_res_block_1_29/StatefulPartitionedCall+^conv_res_block_1_3/StatefulPartitionedCall,^conv_res_block_1_30/StatefulPartitionedCall,^conv_res_block_1_31/StatefulPartitionedCall,^conv_res_block_1_32/StatefulPartitionedCall,^conv_res_block_1_33/StatefulPartitionedCall,^conv_res_block_1_34/StatefulPartitionedCall,^conv_res_block_1_35/StatefulPartitionedCall,^conv_res_block_1_36/StatefulPartitionedCall,^conv_res_block_1_37/StatefulPartitionedCall,^conv_res_block_1_38/StatefulPartitionedCall,^conv_res_block_1_39/StatefulPartitionedCall+^conv_res_block_1_4/StatefulPartitionedCall,^conv_res_block_1_40/StatefulPartitionedCall,^conv_res_block_1_41/StatefulPartitionedCall,^conv_res_block_1_42/StatefulPartitionedCall,^conv_res_block_1_43/StatefulPartitionedCall,^conv_res_block_1_44/StatefulPartitionedCall,^conv_res_block_1_45/StatefulPartitionedCall,^conv_res_block_1_46/StatefulPartitionedCall,^conv_res_block_1_47/StatefulPartitionedCall,^conv_res_block_1_48/StatefulPartitionedCall,^conv_res_block_1_49/StatefulPartitionedCall+^conv_res_block_1_5/StatefulPartitionedCall,^conv_res_block_1_50/StatefulPartitionedCall,^conv_res_block_1_51/StatefulPartitionedCall+^conv_res_block_1_6/StatefulPartitionedCall+^conv_res_block_1_7/StatefulPartitionedCall+^conv_res_block_1_8/StatefulPartitionedCall+^conv_res_block_1_9/StatefulPartitionedCall)^conv_res_block_2/StatefulPartitionedCall)^conv_res_block_3/StatefulPartitionedCall)^conv_res_block_4/StatefulPartitionedCall&^inner_decoder/StatefulPartitionedCall(^inner_decoder_1/StatefulPartitionedCall)^inner_decoder_10/StatefulPartitionedCall)^inner_decoder_11/StatefulPartitionedCall)^inner_decoder_12/StatefulPartitionedCall)^inner_decoder_13/StatefulPartitionedCall)^inner_decoder_14/StatefulPartitionedCall)^inner_decoder_15/StatefulPartitionedCall)^inner_decoder_16/StatefulPartitionedCall)^inner_decoder_17/StatefulPartitionedCall)^inner_decoder_18/StatefulPartitionedCall)^inner_decoder_19/StatefulPartitionedCall(^inner_decoder_2/StatefulPartitionedCall)^inner_decoder_20/StatefulPartitionedCall)^inner_decoder_21/StatefulPartitionedCall)^inner_decoder_22/StatefulPartitionedCall)^inner_decoder_23/StatefulPartitionedCall)^inner_decoder_24/StatefulPartitionedCall)^inner_decoder_25/StatefulPartitionedCall)^inner_decoder_26/StatefulPartitionedCall)^inner_decoder_27/StatefulPartitionedCall)^inner_decoder_28/StatefulPartitionedCall)^inner_decoder_29/StatefulPartitionedCall(^inner_decoder_3/StatefulPartitionedCall)^inner_decoder_30/StatefulPartitionedCall)^inner_decoder_31/StatefulPartitionedCall)^inner_decoder_32/StatefulPartitionedCall)^inner_decoder_33/StatefulPartitionedCall)^inner_decoder_34/StatefulPartitionedCall)^inner_decoder_35/StatefulPartitionedCall)^inner_decoder_36/StatefulPartitionedCall)^inner_decoder_37/StatefulPartitionedCall)^inner_decoder_38/StatefulPartitionedCall)^inner_decoder_39/StatefulPartitionedCall(^inner_decoder_4/StatefulPartitionedCall)^inner_decoder_40/StatefulPartitionedCall)^inner_decoder_41/StatefulPartitionedCall)^inner_decoder_42/StatefulPartitionedCall)^inner_decoder_43/StatefulPartitionedCall)^inner_decoder_44/StatefulPartitionedCall)^inner_decoder_45/StatefulPartitionedCall)^inner_decoder_46/StatefulPartitionedCall)^inner_decoder_47/StatefulPartitionedCall)^inner_decoder_48/StatefulPartitionedCall)^inner_decoder_49/StatefulPartitionedCall(^inner_decoder_5/StatefulPartitionedCall)^inner_decoder_50/StatefulPartitionedCall(^inner_decoder_6/StatefulPartitionedCall(^inner_decoder_7/StatefulPartitionedCall(^inner_decoder_8/StatefulPartitionedCall(^inner_decoder_9/StatefulPartitionedCall&^inner_encoder/StatefulPartitionedCall(^inner_encoder_1/StatefulPartitionedCall(^inner_encoder_2/StatefulPartitionedCall(^inner_encoder_3/StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Identity_102"%
identity_100Identity_100:output:0"%
identity_101Identity_101:output:0"%
identity_102Identity_102:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3:::::::::::::::::::::::::::2P
&conv_res_block/StatefulPartitionedCall&conv_res_block/StatefulPartitionedCall2T
(conv_res_block_1/StatefulPartitionedCall(conv_res_block_1/StatefulPartitionedCall2X
*conv_res_block_1_1/StatefulPartitionedCall*conv_res_block_1_1/StatefulPartitionedCall2Z
+conv_res_block_1_10/StatefulPartitionedCall+conv_res_block_1_10/StatefulPartitionedCall2Z
+conv_res_block_1_11/StatefulPartitionedCall+conv_res_block_1_11/StatefulPartitionedCall2Z
+conv_res_block_1_12/StatefulPartitionedCall+conv_res_block_1_12/StatefulPartitionedCall2Z
+conv_res_block_1_13/StatefulPartitionedCall+conv_res_block_1_13/StatefulPartitionedCall2Z
+conv_res_block_1_14/StatefulPartitionedCall+conv_res_block_1_14/StatefulPartitionedCall2Z
+conv_res_block_1_15/StatefulPartitionedCall+conv_res_block_1_15/StatefulPartitionedCall2Z
+conv_res_block_1_16/StatefulPartitionedCall+conv_res_block_1_16/StatefulPartitionedCall2Z
+conv_res_block_1_17/StatefulPartitionedCall+conv_res_block_1_17/StatefulPartitionedCall2Z
+conv_res_block_1_18/StatefulPartitionedCall+conv_res_block_1_18/StatefulPartitionedCall2Z
+conv_res_block_1_19/StatefulPartitionedCall+conv_res_block_1_19/StatefulPartitionedCall2X
*conv_res_block_1_2/StatefulPartitionedCall*conv_res_block_1_2/StatefulPartitionedCall2Z
+conv_res_block_1_20/StatefulPartitionedCall+conv_res_block_1_20/StatefulPartitionedCall2Z
+conv_res_block_1_21/StatefulPartitionedCall+conv_res_block_1_21/StatefulPartitionedCall2Z
+conv_res_block_1_22/StatefulPartitionedCall+conv_res_block_1_22/StatefulPartitionedCall2Z
+conv_res_block_1_23/StatefulPartitionedCall+conv_res_block_1_23/StatefulPartitionedCall2Z
+conv_res_block_1_24/StatefulPartitionedCall+conv_res_block_1_24/StatefulPartitionedCall2Z
+conv_res_block_1_25/StatefulPartitionedCall+conv_res_block_1_25/StatefulPartitionedCall2Z
+conv_res_block_1_26/StatefulPartitionedCall+conv_res_block_1_26/StatefulPartitionedCall2Z
+conv_res_block_1_27/StatefulPartitionedCall+conv_res_block_1_27/StatefulPartitionedCall2Z
+conv_res_block_1_28/StatefulPartitionedCall+conv_res_block_1_28/StatefulPartitionedCall2Z
+conv_res_block_1_29/StatefulPartitionedCall+conv_res_block_1_29/StatefulPartitionedCall2X
*conv_res_block_1_3/StatefulPartitionedCall*conv_res_block_1_3/StatefulPartitionedCall2Z
+conv_res_block_1_30/StatefulPartitionedCall+conv_res_block_1_30/StatefulPartitionedCall2Z
+conv_res_block_1_31/StatefulPartitionedCall+conv_res_block_1_31/StatefulPartitionedCall2Z
+conv_res_block_1_32/StatefulPartitionedCall+conv_res_block_1_32/StatefulPartitionedCall2Z
+conv_res_block_1_33/StatefulPartitionedCall+conv_res_block_1_33/StatefulPartitionedCall2Z
+conv_res_block_1_34/StatefulPartitionedCall+conv_res_block_1_34/StatefulPartitionedCall2Z
+conv_res_block_1_35/StatefulPartitionedCall+conv_res_block_1_35/StatefulPartitionedCall2Z
+conv_res_block_1_36/StatefulPartitionedCall+conv_res_block_1_36/StatefulPartitionedCall2Z
+conv_res_block_1_37/StatefulPartitionedCall+conv_res_block_1_37/StatefulPartitionedCall2Z
+conv_res_block_1_38/StatefulPartitionedCall+conv_res_block_1_38/StatefulPartitionedCall2Z
+conv_res_block_1_39/StatefulPartitionedCall+conv_res_block_1_39/StatefulPartitionedCall2X
*conv_res_block_1_4/StatefulPartitionedCall*conv_res_block_1_4/StatefulPartitionedCall2Z
+conv_res_block_1_40/StatefulPartitionedCall+conv_res_block_1_40/StatefulPartitionedCall2Z
+conv_res_block_1_41/StatefulPartitionedCall+conv_res_block_1_41/StatefulPartitionedCall2Z
+conv_res_block_1_42/StatefulPartitionedCall+conv_res_block_1_42/StatefulPartitionedCall2Z
+conv_res_block_1_43/StatefulPartitionedCall+conv_res_block_1_43/StatefulPartitionedCall2Z
+conv_res_block_1_44/StatefulPartitionedCall+conv_res_block_1_44/StatefulPartitionedCall2Z
+conv_res_block_1_45/StatefulPartitionedCall+conv_res_block_1_45/StatefulPartitionedCall2Z
+conv_res_block_1_46/StatefulPartitionedCall+conv_res_block_1_46/StatefulPartitionedCall2Z
+conv_res_block_1_47/StatefulPartitionedCall+conv_res_block_1_47/StatefulPartitionedCall2Z
+conv_res_block_1_48/StatefulPartitionedCall+conv_res_block_1_48/StatefulPartitionedCall2Z
+conv_res_block_1_49/StatefulPartitionedCall+conv_res_block_1_49/StatefulPartitionedCall2X
*conv_res_block_1_5/StatefulPartitionedCall*conv_res_block_1_5/StatefulPartitionedCall2Z
+conv_res_block_1_50/StatefulPartitionedCall+conv_res_block_1_50/StatefulPartitionedCall2Z
+conv_res_block_1_51/StatefulPartitionedCall+conv_res_block_1_51/StatefulPartitionedCall2X
*conv_res_block_1_6/StatefulPartitionedCall*conv_res_block_1_6/StatefulPartitionedCall2X
*conv_res_block_1_7/StatefulPartitionedCall*conv_res_block_1_7/StatefulPartitionedCall2X
*conv_res_block_1_8/StatefulPartitionedCall*conv_res_block_1_8/StatefulPartitionedCall2X
*conv_res_block_1_9/StatefulPartitionedCall*conv_res_block_1_9/StatefulPartitionedCall2T
(conv_res_block_2/StatefulPartitionedCall(conv_res_block_2/StatefulPartitionedCall2T
(conv_res_block_3/StatefulPartitionedCall(conv_res_block_3/StatefulPartitionedCall2T
(conv_res_block_4/StatefulPartitionedCall(conv_res_block_4/StatefulPartitionedCall2N
%inner_decoder/StatefulPartitionedCall%inner_decoder/StatefulPartitionedCall2R
'inner_decoder_1/StatefulPartitionedCall'inner_decoder_1/StatefulPartitionedCall2T
(inner_decoder_10/StatefulPartitionedCall(inner_decoder_10/StatefulPartitionedCall2T
(inner_decoder_11/StatefulPartitionedCall(inner_decoder_11/StatefulPartitionedCall2T
(inner_decoder_12/StatefulPartitionedCall(inner_decoder_12/StatefulPartitionedCall2T
(inner_decoder_13/StatefulPartitionedCall(inner_decoder_13/StatefulPartitionedCall2T
(inner_decoder_14/StatefulPartitionedCall(inner_decoder_14/StatefulPartitionedCall2T
(inner_decoder_15/StatefulPartitionedCall(inner_decoder_15/StatefulPartitionedCall2T
(inner_decoder_16/StatefulPartitionedCall(inner_decoder_16/StatefulPartitionedCall2T
(inner_decoder_17/StatefulPartitionedCall(inner_decoder_17/StatefulPartitionedCall2T
(inner_decoder_18/StatefulPartitionedCall(inner_decoder_18/StatefulPartitionedCall2T
(inner_decoder_19/StatefulPartitionedCall(inner_decoder_19/StatefulPartitionedCall2R
'inner_decoder_2/StatefulPartitionedCall'inner_decoder_2/StatefulPartitionedCall2T
(inner_decoder_20/StatefulPartitionedCall(inner_decoder_20/StatefulPartitionedCall2T
(inner_decoder_21/StatefulPartitionedCall(inner_decoder_21/StatefulPartitionedCall2T
(inner_decoder_22/StatefulPartitionedCall(inner_decoder_22/StatefulPartitionedCall2T
(inner_decoder_23/StatefulPartitionedCall(inner_decoder_23/StatefulPartitionedCall2T
(inner_decoder_24/StatefulPartitionedCall(inner_decoder_24/StatefulPartitionedCall2T
(inner_decoder_25/StatefulPartitionedCall(inner_decoder_25/StatefulPartitionedCall2T
(inner_decoder_26/StatefulPartitionedCall(inner_decoder_26/StatefulPartitionedCall2T
(inner_decoder_27/StatefulPartitionedCall(inner_decoder_27/StatefulPartitionedCall2T
(inner_decoder_28/StatefulPartitionedCall(inner_decoder_28/StatefulPartitionedCall2T
(inner_decoder_29/StatefulPartitionedCall(inner_decoder_29/StatefulPartitionedCall2R
'inner_decoder_3/StatefulPartitionedCall'inner_decoder_3/StatefulPartitionedCall2T
(inner_decoder_30/StatefulPartitionedCall(inner_decoder_30/StatefulPartitionedCall2T
(inner_decoder_31/StatefulPartitionedCall(inner_decoder_31/StatefulPartitionedCall2T
(inner_decoder_32/StatefulPartitionedCall(inner_decoder_32/StatefulPartitionedCall2T
(inner_decoder_33/StatefulPartitionedCall(inner_decoder_33/StatefulPartitionedCall2T
(inner_decoder_34/StatefulPartitionedCall(inner_decoder_34/StatefulPartitionedCall2T
(inner_decoder_35/StatefulPartitionedCall(inner_decoder_35/StatefulPartitionedCall2T
(inner_decoder_36/StatefulPartitionedCall(inner_decoder_36/StatefulPartitionedCall2T
(inner_decoder_37/StatefulPartitionedCall(inner_decoder_37/StatefulPartitionedCall2T
(inner_decoder_38/StatefulPartitionedCall(inner_decoder_38/StatefulPartitionedCall2T
(inner_decoder_39/StatefulPartitionedCall(inner_decoder_39/StatefulPartitionedCall2R
'inner_decoder_4/StatefulPartitionedCall'inner_decoder_4/StatefulPartitionedCall2T
(inner_decoder_40/StatefulPartitionedCall(inner_decoder_40/StatefulPartitionedCall2T
(inner_decoder_41/StatefulPartitionedCall(inner_decoder_41/StatefulPartitionedCall2T
(inner_decoder_42/StatefulPartitionedCall(inner_decoder_42/StatefulPartitionedCall2T
(inner_decoder_43/StatefulPartitionedCall(inner_decoder_43/StatefulPartitionedCall2T
(inner_decoder_44/StatefulPartitionedCall(inner_decoder_44/StatefulPartitionedCall2T
(inner_decoder_45/StatefulPartitionedCall(inner_decoder_45/StatefulPartitionedCall2T
(inner_decoder_46/StatefulPartitionedCall(inner_decoder_46/StatefulPartitionedCall2T
(inner_decoder_47/StatefulPartitionedCall(inner_decoder_47/StatefulPartitionedCall2T
(inner_decoder_48/StatefulPartitionedCall(inner_decoder_48/StatefulPartitionedCall2T
(inner_decoder_49/StatefulPartitionedCall(inner_decoder_49/StatefulPartitionedCall2R
'inner_decoder_5/StatefulPartitionedCall'inner_decoder_5/StatefulPartitionedCall2T
(inner_decoder_50/StatefulPartitionedCall(inner_decoder_50/StatefulPartitionedCall2R
'inner_decoder_6/StatefulPartitionedCall'inner_decoder_6/StatefulPartitionedCall2R
'inner_decoder_7/StatefulPartitionedCall'inner_decoder_7/StatefulPartitionedCall2R
'inner_decoder_8/StatefulPartitionedCall'inner_decoder_8/StatefulPartitionedCall2R
'inner_decoder_9/StatefulPartitionedCall'inner_decoder_9/StatefulPartitionedCall2N
%inner_encoder/StatefulPartitionedCall%inner_encoder/StatefulPartitionedCall2R
'inner_encoder_1/StatefulPartitionedCall'inner_encoder_1/StatefulPartitionedCall2R
'inner_encoder_2/StatefulPartitionedCall'inner_encoder_2/StatefulPartitionedCall2R
'inner_encoder_3/StatefulPartitionedCall'inner_encoder_3/StatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


__inference_loss_fn_1_13475325P
Lnetwork_arch_inner_decoder_kernel_regularizer_square_readvariableop_resource
identity
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpLnetwork_arch_inner_decoder_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpí
4network_arch/inner_decoder/kernel/Regularizer/SquareSquareKnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	26
4network_arch/inner_decoder/kernel/Regularizer/Square»
3network_arch/inner_decoder/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       25
3network_arch/inner_decoder/kernel/Regularizer/Const
1network_arch/inner_decoder/kernel/Regularizer/SumSum8network_arch/inner_decoder/kernel/Regularizer/Square:y:0<network_arch/inner_decoder/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/Sum¯
3network_arch/inner_decoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+225
3network_arch/inner_decoder/kernel/Regularizer/mul/x
1network_arch/inner_decoder/kernel/Regularizer/mulMul<network_arch/inner_decoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_decoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/mul¯
3network_arch/inner_decoder/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3network_arch/inner_decoder/kernel/Regularizer/add/x
1network_arch/inner_decoder/kernel/Regularizer/addAddV2<network_arch/inner_decoder/kernel/Regularizer/add/x:output:05network_arch/inner_decoder/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/addx
IdentityIdentity5network_arch/inner_decoder/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ã
R
6__inference_average_pooling1d_5_layer_call_fn_13477564

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Z
fURS
Q__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_134775582
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
h
__inference_loss_fn_7_134777755
1kernel_regularizer_square_readvariableop_resource
identityÈ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/add]
IdentityIdentitykernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ª
~
)__inference_conv1d_layer_call_fn_13477264

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_134772542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ð
»
F__inference_conv1d_5_layer_call_and_return_conditional_losses_13477489

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¿
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
ReluÄ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/adds
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ã
R
6__inference_average_pooling1d_1_layer_call_fn_13477329

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Z
fURS
Q__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_134773232
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
i
__inference_loss_fn_13_134778535
1kernel_regularizer_square_readvariableop_resource
identityÈ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/add]
IdentityIdentitykernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
¯

+__inference_conv1d_2_layer_call_fn_13477364

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_134773542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
÷
m
Q__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_13477458

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsº
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
R
6__inference_average_pooling1d_2_layer_call_fn_13477379

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Z
fURS
Q__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_134773732
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

K__inference_inner_encoder_layer_call_and_return_conditional_losses_13470200

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulê
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpí
4network_arch/inner_encoder/kernel/Regularizer/SquareSquareKnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	26
4network_arch/inner_encoder/kernel/Regularizer/Square»
3network_arch/inner_encoder/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       25
3network_arch/inner_encoder/kernel/Regularizer/Const
1network_arch/inner_encoder/kernel/Regularizer/SumSum8network_arch/inner_encoder/kernel/Regularizer/Square:y:0<network_arch/inner_encoder/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/Sum¯
3network_arch/inner_encoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+225
3network_arch/inner_encoder/kernel/Regularizer/mul/x
1network_arch/inner_encoder/kernel/Regularizer/mulMul<network_arch/inner_encoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_encoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/mul¯
3network_arch/inner_encoder/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3network_arch/inner_encoder/kernel/Regularizer/add/x
1network_arch/inner_encoder/kernel/Regularizer/addAddV2<network_arch/inner_encoder/kernel/Regularizer/add/x:output:05network_arch/inner_encoder/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/addd
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
È
v
0__inference_inner_encoder_layer_call_fn_13475140

inputs
unknown
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_encoder_layer_call_and_return_conditional_losses_134702002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
÷
m
Q__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_13477558

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsº
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
»
F__inference_conv1d_7_layer_call_and_return_conditional_losses_13477589

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1¿
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
ReluÄ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/adds
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ã
µ
L__inference_conv_res_block_layer_call_and_return_conditional_losses_13475270

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim~

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDims~
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/conv1d/ExpandDims/dim¹
conv1d/conv1d/ExpandDims
ExpandDimsExpandDims:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/conv1d/ExpandDimsÍ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimÓ
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/conv1d/ExpandDims_1Ó
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d/conv1d
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d/conv1d/Squeeze¡
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOp©
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/BiasAddr
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/Relu
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dimË
average_pooling1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0)average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
average_pooling1d/ExpandDimsÞ
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
average_pooling1d/AvgPool²
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
average_pooling1d/Squeeze
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_1/conv1d/ExpandDims/dimÍ
conv1d_1/conv1d/ExpandDims
ExpandDims"average_pooling1d/Squeeze:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_1/conv1d/ExpandDimsÓ
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimÛ
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_1/conv1d/ExpandDims_1Ú
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv1d_1/conv1d¤
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
conv1d_1/conv1d/Squeeze§
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp°
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_1/Relu
"average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_1/ExpandDims/dimÒ
average_pooling1d_1/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0+average_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
average_pooling1d_1/ExpandDimsä
average_pooling1d_1/AvgPoolAvgPool'average_pooling1d_1/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_1/AvgPool¸
average_pooling1d_1/SqueezeSqueeze$average_pooling1d_1/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2
average_pooling1d_1/Squeeze
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_2/conv1d/ExpandDims/dimÏ
conv1d_2/conv1d/ExpandDims
ExpandDims$average_pooling1d_1/Squeeze:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_2/conv1d/ExpandDimsÓ
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimÛ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_2/conv1d/ExpandDims_1Ú
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv1d_2/conv1d¤
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
squeeze_dims
2
conv1d_2/conv1d/Squeeze§
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_2/BiasAdd/ReadVariableOp°
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_2/Relu
"average_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_2/ExpandDims/dimÒ
average_pooling1d_2/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0+average_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2 
average_pooling1d_2/ExpandDimsä
average_pooling1d_2/AvgPoolAvgPool'average_pooling1d_2/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_2/AvgPool¸
average_pooling1d_2/SqueezeSqueeze$average_pooling1d_2/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2
average_pooling1d_2/Squeeze
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_3/conv1d/ExpandDims/dimÏ
conv1d_3/conv1d/ExpandDims
ExpandDims$average_pooling1d_2/Squeeze:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_3/conv1d/ExpandDimsÓ
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dimÛ
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_3/conv1d/ExpandDims_1Ú
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv1d_3/conv1d¤
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
conv1d_3/conv1d/Squeeze§
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp°
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_3/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshapeconv1d_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape¡
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddh
addAddV2inputsdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
Knetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02M
Knetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Square/ReadVariableOp
<network_arch/conv_res_block/conv1d/kernel/Regularizer/SquareSquareSnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2>
<network_arch/conv_res_block/conv1d/kernel/Regularizer/SquareÏ
;network_arch/conv_res_block/conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2=
;network_arch/conv_res_block/conv1d/kernel/Regularizer/Const¦
9network_arch/conv_res_block/conv1d/kernel/Regularizer/SumSum@network_arch/conv_res_block/conv1d/kernel/Regularizer/Square:y:0Dnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2;
9network_arch/conv_res_block/conv1d/kernel/Regularizer/Sum¿
;network_arch/conv_res_block/conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22=
;network_arch/conv_res_block/conv1d/kernel/Regularizer/mul/x¨
9network_arch/conv_res_block/conv1d/kernel/Regularizer/mulMulDnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/mul/x:output:0Bnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2;
9network_arch/conv_res_block/conv1d/kernel/Regularizer/mul¿
;network_arch/conv_res_block/conv1d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2=
;network_arch/conv_res_block/conv1d/kernel/Regularizer/add/x¥
9network_arch/conv_res_block/conv1d/kernel/Regularizer/addAddV2Dnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/add/x:output:0=network_arch/conv_res_block/conv1d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2;
9network_arch/conv_res_block/conv1d/kernel/Regularizer/add
Mnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02O
Mnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Square/ReadVariableOp
>network_arch/conv_res_block/conv1d_1/kernel/Regularizer/SquareSquareUnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2@
>network_arch/conv_res_block/conv1d_1/kernel/Regularizer/SquareÓ
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/Const®
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/SumSumBnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Square:y:0Fnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/SumÃ
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22?
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mul/x°
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mulMulFnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/mul/x:output:0Dnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mulÃ
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/add/x­
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/addAddV2Fnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/add/x:output:0?network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/add
Mnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02O
Mnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Square/ReadVariableOp
>network_arch/conv_res_block/conv1d_2/kernel/Regularizer/SquareSquareUnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2@
>network_arch/conv_res_block/conv1d_2/kernel/Regularizer/SquareÓ
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/Const®
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/SumSumBnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Square:y:0Fnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/SumÃ
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22?
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mul/x°
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mulMulFnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/mul/x:output:0Dnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mulÃ
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/add/x­
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/addAddV2Fnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/add/x:output:0?network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/add
Mnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02O
Mnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Square/ReadVariableOp
>network_arch/conv_res_block/conv1d_3/kernel/Regularizer/SquareSquareUnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2@
>network_arch/conv_res_block/conv1d_3/kernel/Regularizer/SquareÓ
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/Const®
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/SumSumBnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Square:y:0Fnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/SumÃ
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22?
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mul/x°
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mulMulFnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/mul/x:output:0Dnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mulÃ
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/add/x­
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/addAddV2Fnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/add/x:output:0?network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/addÿ
Jnetwork_arch/conv_res_block/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02L
Jnetwork_arch/conv_res_block/dense/kernel/Regularizer/Square/ReadVariableOp
;network_arch/conv_res_block/dense/kernel/Regularizer/SquareSquareRnetwork_arch/conv_res_block/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2=
;network_arch/conv_res_block/dense/kernel/Regularizer/SquareÉ
:network_arch/conv_res_block/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2<
:network_arch/conv_res_block/dense/kernel/Regularizer/Const¢
8network_arch/conv_res_block/dense/kernel/Regularizer/SumSum?network_arch/conv_res_block/dense/kernel/Regularizer/Square:y:0Cnetwork_arch/conv_res_block/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2:
8network_arch/conv_res_block/dense/kernel/Regularizer/Sum½
:network_arch/conv_res_block/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22<
:network_arch/conv_res_block/dense/kernel/Regularizer/mul/x¤
8network_arch/conv_res_block/dense/kernel/Regularizer/mulMulCnetwork_arch/conv_res_block/dense/kernel/Regularizer/mul/x:output:0Anetwork_arch/conv_res_block/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2:
8network_arch/conv_res_block/dense/kernel/Regularizer/mul½
:network_arch/conv_res_block/dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2<
:network_arch/conv_res_block/dense/kernel/Regularizer/add/x¡
8network_arch/conv_res_block/dense/kernel/Regularizer/addAddV2Cnetwork_arch/conv_res_block/dense/kernel/Regularizer/add/x:output:0<network_arch/conv_res_block/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2:
8network_arch/conv_res_block/dense/kernel/Regularizer/add
Lnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02N
Lnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Square/ReadVariableOp
=network_arch/conv_res_block/dense_1/kernel/Regularizer/SquareSquareTnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2?
=network_arch/conv_res_block/dense_1/kernel/Regularizer/SquareÍ
<network_arch/conv_res_block/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2>
<network_arch/conv_res_block/dense_1/kernel/Regularizer/Constª
:network_arch/conv_res_block/dense_1/kernel/Regularizer/SumSumAnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Square:y:0Enetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2<
:network_arch/conv_res_block/dense_1/kernel/Regularizer/SumÁ
<network_arch/conv_res_block/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22>
<network_arch/conv_res_block/dense_1/kernel/Regularizer/mul/x¬
:network_arch/conv_res_block/dense_1/kernel/Regularizer/mulMulEnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/mul/x:output:0Cnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2<
:network_arch/conv_res_block/dense_1/kernel/Regularizer/mulÁ
<network_arch/conv_res_block/dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2>
<network_arch/conv_res_block/dense_1/kernel/Regularizer/add/x©
:network_arch/conv_res_block/dense_1/kernel/Regularizer/addAddV2Enetwork_arch/conv_res_block/dense_1/kernel/Regularizer/add/x:output:0>network_arch/conv_res_block/dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2<
:network_arch/conv_res_block/dense_1/kernel/Regularizer/add\
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ:::::::::::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¯

+__inference_conv1d_7_layer_call_fn_13477599

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv1d_7_layer_call_and_return_conditional_losses_134775892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ç
ã
#__inference__wrapped_model_13477052
input_1
network_arch_13476992
network_arch_13476994
network_arch_13476996
network_arch_13476998
network_arch_13477000
network_arch_13477002
network_arch_13477004
network_arch_13477006
network_arch_13477008
network_arch_13477010
network_arch_13477012
network_arch_13477014
network_arch_13477016
network_arch_13477018
network_arch_13477020
network_arch_13477022
network_arch_13477024
network_arch_13477026
network_arch_13477028
network_arch_13477030
network_arch_13477032
network_arch_13477034
network_arch_13477036
network_arch_13477038
network_arch_13477040
network_arch_13477042
network_arch_13477044
identity

identity_1

identity_2¢$network_arch/StatefulPartitionedCall
$network_arch/StatefulPartitionedCallStatefulPartitionedCallinput_1network_arch_13476992network_arch_13476994network_arch_13476996network_arch_13476998network_arch_13477000network_arch_13477002network_arch_13477004network_arch_13477006network_arch_13477008network_arch_13477010network_arch_13477012network_arch_13477014network_arch_13477016network_arch_13477018network_arch_13477020network_arch_13477022network_arch_13477024network_arch_13477026network_arch_13477028network_arch_13477030network_arch_13477032network_arch_13477034network_arch_13477036network_arch_13477038network_arch_13477040network_arch_13477042network_arch_13477044*'
Tin 
2*
Tout
2*\
_output_shapesJ
H:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ2*=
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*4
f/R-
+__inference_restored_function_body_134769912&
$network_arch/StatefulPartitionedCall­
IdentityIdentity-network_arch/StatefulPartitionedCall:output:0%^network_arch/StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity±

Identity_1Identity-network_arch/StatefulPartitionedCall:output:1%^network_arch/StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity_1±

Identity_2Identity-network_arch/StatefulPartitionedCall:output:2%^network_arch/StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3:::::::::::::::::::::::::::2L
$network_arch/StatefulPartitionedCall$network_arch/StatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ð
»
F__inference_conv1d_4_layer_call_and_return_conditional_losses_13477439

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¿
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
ReluÄ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/adds
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ç
£
&__inference_signature_wrapper_13477229
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25
identity

identity_1

identity_2¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*'
Tin 
2*
Tout
2*\
_output_shapesJ
H:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ2*=
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*,
f'R%
#__inference__wrapped_model_134770522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3:::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
é
i
__inference_loss_fn_11_134778275
1kernel_regularizer_square_readvariableop_resource
identityÊ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*"
_output_shapes
: @*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/add]
IdentityIdentitykernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ý

3__inference_conv_res_block_1_layer_call_fn_13470154

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_134701372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
õ
k
O__inference_average_pooling1d_layer_call_and_return_conditional_losses_13477273

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsº
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
»
F__inference_conv1d_1_layer_call_and_return_conditional_losses_13477304

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¿
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
ReluÄ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/adds
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ð
»
F__inference_conv1d_6_layer_call_and_return_conditional_losses_13477539

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1¿
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
ReluÄ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/adds
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


__inference_loss_fn_0_13475349P
Lnetwork_arch_inner_encoder_kernel_regularizer_square_readvariableop_resource
identity
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpLnetwork_arch_inner_encoder_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpí
4network_arch/inner_encoder/kernel/Regularizer/SquareSquareKnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	26
4network_arch/inner_encoder/kernel/Regularizer/Square»
3network_arch/inner_encoder/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       25
3network_arch/inner_encoder/kernel/Regularizer/Const
1network_arch/inner_encoder/kernel/Regularizer/SumSum8network_arch/inner_encoder/kernel/Regularizer/Square:y:0<network_arch/inner_encoder/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/Sum¯
3network_arch/inner_encoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+225
3network_arch/inner_encoder/kernel/Regularizer/mul/x
1network_arch/inner_encoder/kernel/Regularizer/mulMul<network_arch/inner_encoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_encoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/mul¯
3network_arch/inner_encoder/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3network_arch/inner_encoder/kernel/Regularizer/add/x
1network_arch/inner_encoder/kernel/Regularizer/addAddV2<network_arch/inner_encoder/kernel/Regularizer/add/x:output:05network_arch/inner_encoder/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/addx
IdentityIdentity5network_arch/inner_encoder/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ð
»
F__inference_conv1d_2_layer_call_and_return_conditional_losses_13477354

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1¿
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
ReluÄ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/adds
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
á
i
__inference_loss_fn_12_134778405
1kernel_regularizer_square_readvariableop_resource
identityÈ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/add]
IdentityIdentitykernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Î
µ
L__inference_conv_res_block_layer_call_and_return_conditional_losses_13469166

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim~

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDims~
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/conv1d/ExpandDims/dim¹
conv1d/conv1d/ExpandDims
ExpandDimsExpandDims:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/conv1d/ExpandDimsÍ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimÓ
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/conv1d/ExpandDims_1Ó
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d/conv1d
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d/conv1d/Squeeze¡
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOp©
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/BiasAddr
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/Relu
conv1d/IdentityIdentityconv1d/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/Identity
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dimÊ
average_pooling1d/ExpandDims
ExpandDimsconv1d/Identity:output:0)average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
average_pooling1d/ExpandDimsÞ
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
average_pooling1d/AvgPool²
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
average_pooling1d/Squeeze
average_pooling1d/IdentityIdentity"average_pooling1d/Squeeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
average_pooling1d/Identity
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_1/conv1d/ExpandDims/dimÎ
conv1d_1/conv1d/ExpandDims
ExpandDims#average_pooling1d/Identity:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_1/conv1d/ExpandDimsÓ
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimÛ
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_1/conv1d/ExpandDims_1Ú
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv1d_1/conv1d¤
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
conv1d_1/conv1d/Squeeze§
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp°
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_1/Relu
conv1d_1/IdentityIdentityconv1d_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_1/Identity
"average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_1/ExpandDims/dimÑ
average_pooling1d_1/ExpandDims
ExpandDimsconv1d_1/Identity:output:0+average_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
average_pooling1d_1/ExpandDimsä
average_pooling1d_1/AvgPoolAvgPool'average_pooling1d_1/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_1/AvgPool¸
average_pooling1d_1/SqueezeSqueeze$average_pooling1d_1/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2
average_pooling1d_1/Squeeze¤
average_pooling1d_1/IdentityIdentity$average_pooling1d_1/Squeeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
average_pooling1d_1/Identity
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_2/conv1d/ExpandDims/dimÐ
conv1d_2/conv1d/ExpandDims
ExpandDims%average_pooling1d_1/Identity:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_2/conv1d/ExpandDimsÓ
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimÛ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_2/conv1d/ExpandDims_1Ú
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv1d_2/conv1d¤
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
squeeze_dims
2
conv1d_2/conv1d/Squeeze§
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_2/BiasAdd/ReadVariableOp°
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_2/Relu
conv1d_2/IdentityIdentityconv1d_2/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_2/Identity
"average_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_2/ExpandDims/dimÑ
average_pooling1d_2/ExpandDims
ExpandDimsconv1d_2/Identity:output:0+average_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2 
average_pooling1d_2/ExpandDimsä
average_pooling1d_2/AvgPoolAvgPool'average_pooling1d_2/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_2/AvgPool¸
average_pooling1d_2/SqueezeSqueeze$average_pooling1d_2/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2
average_pooling1d_2/Squeeze¤
average_pooling1d_2/IdentityIdentity$average_pooling1d_2/Squeeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
average_pooling1d_2/Identity
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_3/conv1d/ExpandDims/dimÐ
conv1d_3/conv1d/ExpandDims
ExpandDims%average_pooling1d_2/Identity:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_3/conv1d/ExpandDimsÓ
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dimÛ
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_3/conv1d/ExpandDims_1Ú
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv1d_3/conv1d¤
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
conv1d_3/conv1d/Squeeze§
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp°
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_3/Relu
conv1d_3/IdentityIdentityconv1d_3/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv1d_3/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshapeconv1d_3/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape}
flatten/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Identity¡
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Reluy
dense/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Identity§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAdd}
dense_1/IdentityIdentitydense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Identityi
addAddV2inputsdense_1/Identity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
Knetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02M
Knetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Square/ReadVariableOp
<network_arch/conv_res_block/conv1d/kernel/Regularizer/SquareSquareSnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2>
<network_arch/conv_res_block/conv1d/kernel/Regularizer/SquareÏ
;network_arch/conv_res_block/conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2=
;network_arch/conv_res_block/conv1d/kernel/Regularizer/Const¦
9network_arch/conv_res_block/conv1d/kernel/Regularizer/SumSum@network_arch/conv_res_block/conv1d/kernel/Regularizer/Square:y:0Dnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2;
9network_arch/conv_res_block/conv1d/kernel/Regularizer/Sum¿
;network_arch/conv_res_block/conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22=
;network_arch/conv_res_block/conv1d/kernel/Regularizer/mul/x¨
9network_arch/conv_res_block/conv1d/kernel/Regularizer/mulMulDnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/mul/x:output:0Bnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2;
9network_arch/conv_res_block/conv1d/kernel/Regularizer/mul¿
;network_arch/conv_res_block/conv1d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2=
;network_arch/conv_res_block/conv1d/kernel/Regularizer/add/x¥
9network_arch/conv_res_block/conv1d/kernel/Regularizer/addAddV2Dnetwork_arch/conv_res_block/conv1d/kernel/Regularizer/add/x:output:0=network_arch/conv_res_block/conv1d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2;
9network_arch/conv_res_block/conv1d/kernel/Regularizer/add
Mnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02O
Mnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Square/ReadVariableOp
>network_arch/conv_res_block/conv1d_1/kernel/Regularizer/SquareSquareUnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2@
>network_arch/conv_res_block/conv1d_1/kernel/Regularizer/SquareÓ
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/Const®
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/SumSumBnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Square:y:0Fnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/SumÃ
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22?
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mul/x°
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mulMulFnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/mul/x:output:0Dnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mulÃ
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=network_arch/conv_res_block/conv1d_1/kernel/Regularizer/add/x­
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/addAddV2Fnetwork_arch/conv_res_block/conv1d_1/kernel/Regularizer/add/x:output:0?network_arch/conv_res_block/conv1d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_1/kernel/Regularizer/add
Mnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02O
Mnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Square/ReadVariableOp
>network_arch/conv_res_block/conv1d_2/kernel/Regularizer/SquareSquareUnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2@
>network_arch/conv_res_block/conv1d_2/kernel/Regularizer/SquareÓ
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/Const®
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/SumSumBnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Square:y:0Fnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/SumÃ
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22?
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mul/x°
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mulMulFnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/mul/x:output:0Dnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mulÃ
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=network_arch/conv_res_block/conv1d_2/kernel/Regularizer/add/x­
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/addAddV2Fnetwork_arch/conv_res_block/conv1d_2/kernel/Regularizer/add/x:output:0?network_arch/conv_res_block/conv1d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_2/kernel/Regularizer/add
Mnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02O
Mnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Square/ReadVariableOp
>network_arch/conv_res_block/conv1d_3/kernel/Regularizer/SquareSquareUnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2@
>network_arch/conv_res_block/conv1d_3/kernel/Regularizer/SquareÓ
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/Const®
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/SumSumBnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Square:y:0Fnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/SumÃ
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22?
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mul/x°
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mulMulFnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/mul/x:output:0Dnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mulÃ
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=network_arch/conv_res_block/conv1d_3/kernel/Regularizer/add/x­
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/addAddV2Fnetwork_arch/conv_res_block/conv1d_3/kernel/Regularizer/add/x:output:0?network_arch/conv_res_block/conv1d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2=
;network_arch/conv_res_block/conv1d_3/kernel/Regularizer/addÿ
Jnetwork_arch/conv_res_block/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02L
Jnetwork_arch/conv_res_block/dense/kernel/Regularizer/Square/ReadVariableOp
;network_arch/conv_res_block/dense/kernel/Regularizer/SquareSquareRnetwork_arch/conv_res_block/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2=
;network_arch/conv_res_block/dense/kernel/Regularizer/SquareÉ
:network_arch/conv_res_block/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2<
:network_arch/conv_res_block/dense/kernel/Regularizer/Const¢
8network_arch/conv_res_block/dense/kernel/Regularizer/SumSum?network_arch/conv_res_block/dense/kernel/Regularizer/Square:y:0Cnetwork_arch/conv_res_block/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2:
8network_arch/conv_res_block/dense/kernel/Regularizer/Sum½
:network_arch/conv_res_block/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22<
:network_arch/conv_res_block/dense/kernel/Regularizer/mul/x¤
8network_arch/conv_res_block/dense/kernel/Regularizer/mulMulCnetwork_arch/conv_res_block/dense/kernel/Regularizer/mul/x:output:0Anetwork_arch/conv_res_block/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2:
8network_arch/conv_res_block/dense/kernel/Regularizer/mul½
:network_arch/conv_res_block/dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2<
:network_arch/conv_res_block/dense/kernel/Regularizer/add/x¡
8network_arch/conv_res_block/dense/kernel/Regularizer/addAddV2Cnetwork_arch/conv_res_block/dense/kernel/Regularizer/add/x:output:0<network_arch/conv_res_block/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2:
8network_arch/conv_res_block/dense/kernel/Regularizer/add
Lnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02N
Lnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Square/ReadVariableOp
=network_arch/conv_res_block/dense_1/kernel/Regularizer/SquareSquareTnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2?
=network_arch/conv_res_block/dense_1/kernel/Regularizer/SquareÍ
<network_arch/conv_res_block/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2>
<network_arch/conv_res_block/dense_1/kernel/Regularizer/Constª
:network_arch/conv_res_block/dense_1/kernel/Regularizer/SumSumAnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Square:y:0Enetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2<
:network_arch/conv_res_block/dense_1/kernel/Regularizer/SumÁ
<network_arch/conv_res_block/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22>
<network_arch/conv_res_block/dense_1/kernel/Regularizer/mul/x¬
:network_arch/conv_res_block/dense_1/kernel/Regularizer/mulMulEnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/mul/x:output:0Cnetwork_arch/conv_res_block/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2<
:network_arch/conv_res_block/dense_1/kernel/Regularizer/mulÁ
<network_arch/conv_res_block/dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2>
<network_arch/conv_res_block/dense_1/kernel/Regularizer/add/x©
:network_arch/conv_res_block/dense_1/kernel/Regularizer/addAddV2Enetwork_arch/conv_res_block/dense_1/kernel/Regularizer/add/x:output:0>network_arch/conv_res_block/dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2<
:network_arch/conv_res_block/dense_1/kernel/Regularizer/add\
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ:::::::::::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ÎK
Á
!__inference__traced_save_13478059
file_prefix'
#savev2_variable_read_readvariableop@
<savev2_network_arch_inner_encoder_kernel_read_readvariableop@
<savev2_network_arch_inner_decoder_kernel_read_readvariableopH
Dsavev2_network_arch_conv_res_block_conv1d_kernel_read_readvariableopF
Bsavev2_network_arch_conv_res_block_conv1d_bias_read_readvariableopJ
Fsavev2_network_arch_conv_res_block_conv1d_1_kernel_read_readvariableopH
Dsavev2_network_arch_conv_res_block_conv1d_1_bias_read_readvariableopJ
Fsavev2_network_arch_conv_res_block_conv1d_2_kernel_read_readvariableopH
Dsavev2_network_arch_conv_res_block_conv1d_2_bias_read_readvariableopJ
Fsavev2_network_arch_conv_res_block_conv1d_3_kernel_read_readvariableopH
Dsavev2_network_arch_conv_res_block_conv1d_3_bias_read_readvariableopG
Csavev2_network_arch_conv_res_block_dense_kernel_read_readvariableopE
Asavev2_network_arch_conv_res_block_dense_bias_read_readvariableopI
Esavev2_network_arch_conv_res_block_dense_1_kernel_read_readvariableopG
Csavev2_network_arch_conv_res_block_dense_1_bias_read_readvariableopL
Hsavev2_network_arch_conv_res_block_1_conv1d_4_kernel_read_readvariableopJ
Fsavev2_network_arch_conv_res_block_1_conv1d_4_bias_read_readvariableopL
Hsavev2_network_arch_conv_res_block_1_conv1d_5_kernel_read_readvariableopJ
Fsavev2_network_arch_conv_res_block_1_conv1d_5_bias_read_readvariableopL
Hsavev2_network_arch_conv_res_block_1_conv1d_6_kernel_read_readvariableopJ
Fsavev2_network_arch_conv_res_block_1_conv1d_6_bias_read_readvariableopL
Hsavev2_network_arch_conv_res_block_1_conv1d_7_kernel_read_readvariableopJ
Fsavev2_network_arch_conv_res_block_1_conv1d_7_bias_read_readvariableopK
Gsavev2_network_arch_conv_res_block_1_dense_2_kernel_read_readvariableopI
Esavev2_network_arch_conv_res_block_1_dense_2_bias_read_readvariableopK
Gsavev2_network_arch_conv_res_block_1_dense_3_kernel_read_readvariableopI
Esavev2_network_arch_conv_res_block_1_dense_3_bias_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ed341c37364d4773aff254facabef5b1/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÑ	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ã
valueÙBÖBL/.ATTRIBUTES/VARIABLE_VALUEB/inner_encoder/kernel/.ATTRIBUTES/VARIABLE_VALUEB/inner_decoder/kernel/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names¾
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop<savev2_network_arch_inner_encoder_kernel_read_readvariableop<savev2_network_arch_inner_decoder_kernel_read_readvariableopDsavev2_network_arch_conv_res_block_conv1d_kernel_read_readvariableopBsavev2_network_arch_conv_res_block_conv1d_bias_read_readvariableopFsavev2_network_arch_conv_res_block_conv1d_1_kernel_read_readvariableopDsavev2_network_arch_conv_res_block_conv1d_1_bias_read_readvariableopFsavev2_network_arch_conv_res_block_conv1d_2_kernel_read_readvariableopDsavev2_network_arch_conv_res_block_conv1d_2_bias_read_readvariableopFsavev2_network_arch_conv_res_block_conv1d_3_kernel_read_readvariableopDsavev2_network_arch_conv_res_block_conv1d_3_bias_read_readvariableopCsavev2_network_arch_conv_res_block_dense_kernel_read_readvariableopAsavev2_network_arch_conv_res_block_dense_bias_read_readvariableopEsavev2_network_arch_conv_res_block_dense_1_kernel_read_readvariableopCsavev2_network_arch_conv_res_block_dense_1_bias_read_readvariableopHsavev2_network_arch_conv_res_block_1_conv1d_4_kernel_read_readvariableopFsavev2_network_arch_conv_res_block_1_conv1d_4_bias_read_readvariableopHsavev2_network_arch_conv_res_block_1_conv1d_5_kernel_read_readvariableopFsavev2_network_arch_conv_res_block_1_conv1d_5_bias_read_readvariableopHsavev2_network_arch_conv_res_block_1_conv1d_6_kernel_read_readvariableopFsavev2_network_arch_conv_res_block_1_conv1d_6_bias_read_readvariableopHsavev2_network_arch_conv_res_block_1_conv1d_7_kernel_read_readvariableopFsavev2_network_arch_conv_res_block_1_conv1d_7_bias_read_readvariableopGsavev2_network_arch_conv_res_block_1_dense_2_kernel_read_readvariableopEsavev2_network_arch_conv_res_block_1_dense_2_bias_read_readvariableopGsavev2_network_arch_conv_res_block_1_dense_3_kernel_read_readvariableopEsavev2_network_arch_conv_res_block_1_dense_3_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *)
dtypes
22
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesÏ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ã
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*¥
_input_shapes
: ::	:	::::: : : @:@:
::
:::::: : : @:@:
::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::%!

_output_shapes
:	:%!

_output_shapes
:	:($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
: : 	

_output_shapes
: :(
$
"
_output_shapes
: @: 

_output_shapes
:@:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
: @: 

_output_shapes
:@:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: 
à
h
__inference_loss_fn_6_134777625
1kernel_regularizer_square_readvariableop_resource
identityÈ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wÌ+22
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muly
kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
kernel/Regularizer/add/x
kernel/Regularizer/addAddV2!kernel/Regularizer/add/x:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
kernel/Regularizer/add]
IdentityIdentitykernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: "¯L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_default§
@
input_15
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ3A
output_15
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ3A
output_25
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿ3A
output_35
StatefulPartitionedCall:2ÿÿÿÿÿÿÿÿÿ2tensorflow/serving/predict:ã
â	
outer_encoder
outer_decoder
inner_encoder
L
inner_decoder
inner_loss_weights
	optimizer
loss
	
signatures

regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
_default_save_signature
__call__"
_tf_keras_model{"class_name": "NetworkArch", "name": "network_arch", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "NetworkArch"}, "training_config": {"loss": [{"class_name": "RelMSE", "config": {"reduction": "auto", "name": null}}, {"class_name": "RelMSE", "config": {"reduction": "auto", "name": null}}, {"class_name": "RelMSE", "config": {"reduction": "auto", "name": null}}], "metrics": null, "weighted_metrics": null, "loss_weights": [1, 1, 1], "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.00046222686069086194, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
®
conv_layers
dense_layers
	variables
regularization_losses
trainable_variables
	keras_api
+ &call_and_return_all_conditional_losses
¡__call__"ú
_tf_keras_layerà{"class_name": "ConvResBlock", "name": "conv_res_block", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"layer was saved without config": true}}
°
conv_layers
dense_layers
	variables
regularization_losses
trainable_variables
	keras_api
+¢&call_and_return_all_conditional_losses
£__call__"ü
_tf_keras_layerâ{"class_name": "ConvResBlock", "name": "conv_res_block_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"layer was saved without config": true}}
§

kernel
	variables
regularization_losses
trainable_variables
	keras_api
+¤&call_and_return_all_conditional_losses
¥__call__"
_tf_keras_layerð{"class_name": "Dense", "name": "inner_encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "inner_encoder", "trainable": true, "dtype": "float32", "units": 21, "activation": "linear", "use_bias": false, "kernel_initializer": "identity_init", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
:2Variable
§

kernel
 	variables
!regularization_losses
"trainable_variables
#	keras_api
+¦&call_and_return_all_conditional_losses
§__call__"
_tf_keras_layerð{"class_name": "Dense", "name": "inner_decoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "inner_decoder", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": false, "kernel_initializer": "identity_init", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 21}}}}
 "
trackable_list_wrapper
"
	optimizer
 "
trackable_list_wrapper
-
¨serving_default"
signature_map
0
©0
ª1"
trackable_list_wrapper
î
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719
820
921
:22
;23
24
25
26"
trackable_list_wrapper
î
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719
820
921
:22
;23
24
25
26"
trackable_list_wrapper
Î
<layer_regularization_losses
=non_trainable_variables
>metrics

regularization_losses
	variables
?layer_metrics

@layers
trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Q
A0
B1
C2
D3
E4
F5
G6"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
v
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11"
trackable_list_wrapper
P
«0
¬1
­2
®3
¯4
°5"
trackable_list_wrapper
v
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11"
trackable_list_wrapper
°
Klayer_regularization_losses
Lnon_trainable_variables
Mmetrics
	variables
regularization_losses
Nlayer_metrics

Olayers
trainable_variables
¡__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
Q
P0
Q1
R2
S3
T4
U5
V6"
trackable_list_wrapper
5
W0
X1
Y2"
trackable_list_wrapper
v
00
11
22
33
44
55
66
77
88
99
:10
;11"
trackable_list_wrapper
P
±0
²1
³2
´3
µ4
¶5"
trackable_list_wrapper
v
00
11
22
33
44
55
66
77
88
99
:10
;11"
trackable_list_wrapper
°
Zlayer_regularization_losses
[non_trainable_variables
\metrics
	variables
regularization_losses
]layer_metrics

^layers
trainable_variables
£__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
4:2	2!network_arch/inner_encoder/kernel
'
0"
trackable_list_wrapper
(
©0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
_layer_regularization_losses
`non_trainable_variables
ametrics
	variables
regularization_losses
blayer_metrics

clayers
trainable_variables
¥__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
4:2	2!network_arch/inner_decoder/kernel
'
0"
trackable_list_wrapper
(
ª0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
dlayer_regularization_losses
enon_trainable_variables
fmetrics
 	variables
!regularization_losses
glayer_metrics

hlayers
"trainable_variables
§__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
?:=2)network_arch/conv_res_block/conv1d/kernel
5:32'network_arch/conv_res_block/conv1d/bias
A:?2+network_arch/conv_res_block/conv1d_1/kernel
7:52)network_arch/conv_res_block/conv1d_1/bias
A:? 2+network_arch/conv_res_block/conv1d_2/kernel
7:5 2)network_arch/conv_res_block/conv1d_2/bias
A:? @2+network_arch/conv_res_block/conv1d_3/kernel
7:5@2)network_arch/conv_res_block/conv1d_3/bias
<::
2(network_arch/conv_res_block/dense/kernel
5:32&network_arch/conv_res_block/dense/bias
>:<
2*network_arch/conv_res_block/dense_1/kernel
7:52(network_arch/conv_res_block/dense_1/bias
C:A2-network_arch/conv_res_block_1/conv1d_4/kernel
9:72+network_arch/conv_res_block_1/conv1d_4/bias
C:A2-network_arch/conv_res_block_1/conv1d_5/kernel
9:72+network_arch/conv_res_block_1/conv1d_5/bias
C:A 2-network_arch/conv_res_block_1/conv1d_6/kernel
9:7 2+network_arch/conv_res_block_1/conv1d_6/bias
C:A @2-network_arch/conv_res_block_1/conv1d_7/kernel
9:7@2+network_arch/conv_res_block_1/conv1d_7/bias
@:>
2,network_arch/conv_res_block_1/dense_2/kernel
9:72*network_arch/conv_res_block_1/dense_2/bias
@:>
2,network_arch/conv_res_block_1/dense_3/kernel
9:72*network_arch/conv_res_block_1/dense_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
¼


$kernel
%bias
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
+·&call_and_return_all_conditional_losses
¸__call__"	
_tf_keras_layerû{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128, 1]}}
à
m	variables
nregularization_losses
otrainable_variables
p	keras_api
+¹&call_and_return_all_conditional_losses
º__call__"Ï
_tf_keras_layerµ{"class_name": "AveragePooling1D", "name": "average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
À


&kernel
'bias
q	variables
rregularization_losses
strainable_variables
t	keras_api
+»&call_and_return_all_conditional_losses
¼__call__"	
_tf_keras_layerÿ{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 64, 8]}}
ä
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
+½&call_and_return_all_conditional_losses
¾__call__"Ó
_tf_keras_layer¹{"class_name": "AveragePooling1D", "name": "average_pooling1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "average_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Â


(kernel
)bias
y	variables
zregularization_losses
{trainable_variables
|	keras_api
+¿&call_and_return_all_conditional_losses
À__call__"	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 32, 16]}}
å
}	variables
~regularization_losses
trainable_variables
	keras_api
+Á&call_and_return_all_conditional_losses
Â__call__"Ó
_tf_keras_layer¹{"class_name": "AveragePooling1D", "name": "average_pooling1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "average_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Æ


*kernel
+bias
	variables
regularization_losses
trainable_variables
	keras_api
+Ã&call_and_return_all_conditional_losses
Ä__call__"	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 16, 32]}}
Å
	variables
regularization_losses
trainable_variables
	keras_api
+Å&call_and_return_all_conditional_losses
Æ__call__"°
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ß

,kernel
-bias
	variables
regularization_losses
trainable_variables
	keras_api
+Ç&call_and_return_all_conditional_losses
È__call__"´
_tf_keras_layer{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 1024]}}
ã

.kernel
/bias
	variables
regularization_losses
trainable_variables
	keras_api
+É&call_and_return_all_conditional_losses
Ê__call__"¸
_tf_keras_layer{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
f
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9"
trackable_list_wrapper
Ä


0kernel
1bias
	variables
regularization_losses
trainable_variables
	keras_api
+Ë&call_and_return_all_conditional_losses
Ì__call__"	
_tf_keras_layerÿ{"class_name": "Conv1D", "name": "conv1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128, 1]}}
è
	variables
regularization_losses
trainable_variables
	keras_api
+Í&call_and_return_all_conditional_losses
Î__call__"Ó
_tf_keras_layer¹{"class_name": "AveragePooling1D", "name": "average_pooling1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "average_pooling1d_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ä


2kernel
3bias
	variables
regularization_losses
trainable_variables
	keras_api
+Ï&call_and_return_all_conditional_losses
Ð__call__"	
_tf_keras_layerÿ{"class_name": "Conv1D", "name": "conv1d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 64, 8]}}
è
	variables
regularization_losses
trainable_variables
 	keras_api
+Ñ&call_and_return_all_conditional_losses
Ò__call__"Ó
_tf_keras_layer¹{"class_name": "AveragePooling1D", "name": "average_pooling1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "average_pooling1d_4", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Æ


4kernel
5bias
¡	variables
¢regularization_losses
£trainable_variables
¤	keras_api
+Ó&call_and_return_all_conditional_losses
Ô__call__"	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 32, 16]}}
è
¥	variables
¦regularization_losses
§trainable_variables
¨	keras_api
+Õ&call_and_return_all_conditional_losses
Ö__call__"Ó
_tf_keras_layer¹{"class_name": "AveragePooling1D", "name": "average_pooling1d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "average_pooling1d_5", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Æ


6kernel
7bias
©	variables
ªregularization_losses
«trainable_variables
¬	keras_api
+×&call_and_return_all_conditional_losses
Ø__call__"	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 16, 32]}}
É
­	variables
®regularization_losses
¯trainable_variables
°	keras_api
+Ù&call_and_return_all_conditional_losses
Ú__call__"´
_tf_keras_layer{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ã

8kernel
9bias
±	variables
²regularization_losses
³trainable_variables
´	keras_api
+Û&call_and_return_all_conditional_losses
Ü__call__"¸
_tf_keras_layer{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 1024]}}
ã

:kernel
;bias
µ	variables
¶regularization_losses
·trainable_variables
¸	keras_api
+Ý&call_and_return_all_conditional_losses
Þ__call__"¸
_tf_keras_layer{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
f
P0
Q1
R2
S3
T4
U5
V6
W7
X8
Y9"
trackable_list_wrapper
(
©0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
ª0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
(
«0"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
µ
 ¹layer_regularization_losses
ºnon_trainable_variables
»metrics
i	variables
jregularization_losses
¼layer_metrics
½layers
ktrainable_variables
¸__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ¾layer_regularization_losses
¿non_trainable_variables
Àmetrics
m	variables
nregularization_losses
Álayer_metrics
Âlayers
otrainable_variables
º__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
(
¬0"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
µ
 Ãlayer_regularization_losses
Änon_trainable_variables
Åmetrics
q	variables
rregularization_losses
Ælayer_metrics
Çlayers
strainable_variables
¼__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Èlayer_regularization_losses
Énon_trainable_variables
Êmetrics
u	variables
vregularization_losses
Ëlayer_metrics
Ìlayers
wtrainable_variables
¾__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
(
­0"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
µ
 Ílayer_regularization_losses
Înon_trainable_variables
Ïmetrics
y	variables
zregularization_losses
Ðlayer_metrics
Ñlayers
{trainable_variables
À__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Òlayer_regularization_losses
Ónon_trainable_variables
Ômetrics
}	variables
~regularization_losses
Õlayer_metrics
Ölayers
trainable_variables
Â__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
(
®0"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
¸
 ×layer_regularization_losses
Ønon_trainable_variables
Ùmetrics
	variables
regularization_losses
Úlayer_metrics
Ûlayers
trainable_variables
Ä__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 Ülayer_regularization_losses
Ýnon_trainable_variables
Þmetrics
	variables
regularization_losses
ßlayer_metrics
àlayers
trainable_variables
Æ__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
(
¯0"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
¸
 álayer_regularization_losses
ânon_trainable_variables
ãmetrics
	variables
regularization_losses
älayer_metrics
ålayers
trainable_variables
È__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
(
°0"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
¸
 ælayer_regularization_losses
çnon_trainable_variables
èmetrics
	variables
regularization_losses
élayer_metrics
êlayers
trainable_variables
Ê__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
(
±0"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
¸
 ëlayer_regularization_losses
ìnon_trainable_variables
ímetrics
	variables
regularization_losses
îlayer_metrics
ïlayers
trainable_variables
Ì__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 ðlayer_regularization_losses
ñnon_trainable_variables
òmetrics
	variables
regularization_losses
ólayer_metrics
ôlayers
trainable_variables
Î__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
(
²0"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
¸
 õlayer_regularization_losses
önon_trainable_variables
÷metrics
	variables
regularization_losses
ølayer_metrics
ùlayers
trainable_variables
Ð__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 úlayer_regularization_losses
ûnon_trainable_variables
ümetrics
	variables
regularization_losses
ýlayer_metrics
þlayers
trainable_variables
Ò__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
.
40
51"
trackable_list_wrapper
(
³0"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
¸
 ÿlayer_regularization_losses
non_trainable_variables
metrics
¡	variables
¢regularization_losses
layer_metrics
layers
£trainable_variables
Ô__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 layer_regularization_losses
non_trainable_variables
metrics
¥	variables
¦regularization_losses
layer_metrics
layers
§trainable_variables
Ö__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
.
60
71"
trackable_list_wrapper
(
´0"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
¸
 layer_regularization_losses
non_trainable_variables
metrics
©	variables
ªregularization_losses
layer_metrics
layers
«trainable_variables
Ø__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 layer_regularization_losses
non_trainable_variables
metrics
­	variables
®regularization_losses
layer_metrics
layers
¯trainable_variables
Ú__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
.
80
91"
trackable_list_wrapper
(
µ0"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
¸
 layer_regularization_losses
non_trainable_variables
metrics
±	variables
²regularization_losses
layer_metrics
layers
³trainable_variables
Ü__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
.
:0
;1"
trackable_list_wrapper
(
¶0"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
¸
 layer_regularization_losses
non_trainable_variables
metrics
µ	variables
¶regularization_losses
layer_metrics
layers
·trainable_variables
Þ__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
(
«0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
¬0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
­0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
®0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
¯0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
°0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
±0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
²0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
³0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
´0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
µ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
¶0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
2
J__inference_network_arch_layer_call_and_return_conditional_losses_13474990Á
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *+¢(
&#
input_1ÿÿÿÿÿÿÿÿÿ3
æ2ã
#__inference__wrapped_model_13477052»
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *+¢(
&#
input_1ÿÿÿÿÿÿÿÿÿ3
ø2õ
/__inference_network_arch_layer_call_fn_13475134Á
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *+¢(
&#
input_1ÿÿÿÿÿÿÿÿÿ3
ì2é
L__inference_conv_res_block_layer_call_and_return_conditional_losses_13475270
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
1__inference_conv_res_block_layer_call_fn_13469183
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_13468847
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
3__inference_conv_res_block_1_layer_call_fn_13470154
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
K__inference_inner_encoder_layer_call_and_return_conditional_losses_13469014
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
0__inference_inner_encoder_layer_call_fn_13475140
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
K__inference_inner_decoder_layer_call_and_return_conditional_losses_13469411
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
0__inference_inner_decoder_layer_call_fn_13469383
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5B3
&__inference_signature_wrapper_13477229input_1
µ2²
__inference_loss_fn_0_13475349
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_1_13475325
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_2_13477710
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_3_13477723
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_4_13477736
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_5_13477749
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_6_13477762
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_7_13477775
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_8_13477788
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_9_13477801
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
¶2³
__inference_loss_fn_10_13477814
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
¶2³
__inference_loss_fn_11_13477827
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
¶2³
__inference_loss_fn_12_13477840
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
¶2³
__inference_loss_fn_13_13477853
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
2
D__inference_conv1d_layer_call_and_return_conditional_losses_13477254Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
û2ø
)__inference_conv1d_layer_call_fn_13477264Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª2§
O__inference_average_pooling1d_layer_call_and_return_conditional_losses_13477273Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
4__inference_average_pooling1d_layer_call_fn_13477279Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
F__inference_conv1d_1_layer_call_and_return_conditional_losses_13477304Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ý2ú
+__inference_conv1d_1_layer_call_fn_13477314Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¬2©
Q__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_13477323Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
6__inference_average_pooling1d_1_layer_call_fn_13477329Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
F__inference_conv1d_2_layer_call_and_return_conditional_losses_13477354Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ý2ú
+__inference_conv1d_2_layer_call_fn_13477364Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¬2©
Q__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_13477373Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
6__inference_average_pooling1d_2_layer_call_fn_13477379Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
F__inference_conv1d_3_layer_call_and_return_conditional_losses_13477404Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ý2ú
+__inference_conv1d_3_layer_call_fn_13477414Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
F__inference_conv1d_4_layer_call_and_return_conditional_losses_13477439Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ý2ú
+__inference_conv1d_4_layer_call_fn_13477449Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¬2©
Q__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_13477458Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
6__inference_average_pooling1d_3_layer_call_fn_13477464Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
F__inference_conv1d_5_layer_call_and_return_conditional_losses_13477489Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ý2ú
+__inference_conv1d_5_layer_call_fn_13477499Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¬2©
Q__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_13477508Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
6__inference_average_pooling1d_4_layer_call_fn_13477514Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
F__inference_conv1d_6_layer_call_and_return_conditional_losses_13477539Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ý2ú
+__inference_conv1d_6_layer_call_fn_13477549Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¬2©
Q__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_13477558Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
6__inference_average_pooling1d_5_layer_call_fn_13477564Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
F__inference_conv1d_7_layer_call_and_return_conditional_losses_13477589Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ý2ú
+__inference_conv1d_7_layer_call_fn_13477599Ê
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ¢
#__inference__wrapped_model_13477052ú$%&'()*+,-./0123456789:;5¢2
+¢(
&#
input_1ÿÿÿÿÿÿÿÿÿ3
ª "£ª
3
output_1'$
output_1ÿÿÿÿÿÿÿÿÿ3
3
output_2'$
output_2ÿÿÿÿÿÿÿÿÿ3
3
output_3'$
output_3ÿÿÿÿÿÿÿÿÿ2Ú
Q__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_13477323E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ±
6__inference_average_pooling1d_1_layer_call_fn_13477329wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
Q__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_13477373E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ±
6__inference_average_pooling1d_2_layer_call_fn_13477379wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
Q__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_13477458E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ±
6__inference_average_pooling1d_3_layer_call_fn_13477464wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
Q__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_13477508E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ±
6__inference_average_pooling1d_4_layer_call_fn_13477514wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
Q__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_13477558E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ±
6__inference_average_pooling1d_5_layer_call_fn_13477564wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØ
O__inference_average_pooling1d_layer_call_and_return_conditional_losses_13477273E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
4__inference_average_pooling1d_layer_call_fn_13477279wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
F__inference_conv1d_1_layer_call_and_return_conditional_losses_13477304v&'<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
+__inference_conv1d_1_layer_call_fn_13477314i&'<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
F__inference_conv1d_2_layer_call_and_return_conditional_losses_13477354v()<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
+__inference_conv1d_2_layer_call_fn_13477364i()<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ À
F__inference_conv1d_3_layer_call_and_return_conditional_losses_13477404v*+<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
+__inference_conv1d_3_layer_call_fn_13477414i*+<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@À
F__inference_conv1d_4_layer_call_and_return_conditional_losses_13477439v01<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
+__inference_conv1d_4_layer_call_fn_13477449i01<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
F__inference_conv1d_5_layer_call_and_return_conditional_losses_13477489v23<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
+__inference_conv1d_5_layer_call_fn_13477499i23<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
F__inference_conv1d_6_layer_call_and_return_conditional_losses_13477539v45<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
+__inference_conv1d_6_layer_call_fn_13477549i45<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ À
F__inference_conv1d_7_layer_call_and_return_conditional_losses_13477589v67<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
+__inference_conv1d_7_layer_call_fn_13477599i67<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¾
D__inference_conv1d_layer_call_and_return_conditional_losses_13477254v$%<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
)__inference_conv1d_layer_call_fn_13477264i$%<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
N__inference_conv_res_block_1_layer_call_and_return_conditional_losses_13468847h0123456789:;0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
3__inference_conv_res_block_1_layer_call_fn_13470154[0123456789:;0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¸
L__inference_conv_res_block_layer_call_and_return_conditional_losses_13475270h$%&'()*+,-./0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_conv_res_block_layer_call_fn_13469183[$%&'()*+,-./0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
K__inference_inner_decoder_layer_call_and_return_conditional_losses_13469411\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_inner_decoder_layer_call_fn_13469383O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
K__inference_inner_encoder_layer_call_and_return_conditional_losses_13469014\0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_inner_encoder_layer_call_fn_13475140O0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ=
__inference_loss_fn_0_13475349¢

¢ 
ª " >
__inference_loss_fn_10_134778144¢

¢ 
ª " >
__inference_loss_fn_11_134778276¢

¢ 
ª " >
__inference_loss_fn_12_134778408¢

¢ 
ª " >
__inference_loss_fn_13_13477853:¢

¢ 
ª " =
__inference_loss_fn_1_13475325¢

¢ 
ª " =
__inference_loss_fn_2_13477710$¢

¢ 
ª " =
__inference_loss_fn_3_13477723&¢

¢ 
ª " =
__inference_loss_fn_4_13477736(¢

¢ 
ª " =
__inference_loss_fn_5_13477749*¢

¢ 
ª " =
__inference_loss_fn_6_13477762,¢

¢ 
ª " =
__inference_loss_fn_7_13477775.¢

¢ 
ª " =
__inference_loss_fn_8_134777880¢

¢ 
ª " =
__inference_loss_fn_9_134778012¢

¢ 
ª " 
J__inference_network_arch_layer_call_and_return_conditional_losses_13474990Ï$%&'()*+,-./0123456789:;5¢2
+¢(
&#
input_1ÿÿÿÿÿÿÿÿÿ3
ª "y¢v
o¢l
"
0/0ÿÿÿÿÿÿÿÿÿ3
"
0/1ÿÿÿÿÿÿÿÿÿ3
"
0/2ÿÿÿÿÿÿÿÿÿ2
 ó
/__inference_network_arch_layer_call_fn_13475134¿$%&'()*+,-./0123456789:;5¢2
+¢(
&#
input_1ÿÿÿÿÿÿÿÿÿ3
ª "i¢f
 
0ÿÿÿÿÿÿÿÿÿ3
 
1ÿÿÿÿÿÿÿÿÿ3
 
2ÿÿÿÿÿÿÿÿÿ2°
&__inference_signature_wrapper_13477229$%&'()*+,-./0123456789:;@¢=
¢ 
6ª3
1
input_1&#
input_1ÿÿÿÿÿÿÿÿÿ3"£ª
3
output_1'$
output_1ÿÿÿÿÿÿÿÿÿ3
3
output_2'$
output_2ÿÿÿÿÿÿÿÿÿ3
3
output_3'$
output_3ÿÿÿÿÿÿÿÿÿ2