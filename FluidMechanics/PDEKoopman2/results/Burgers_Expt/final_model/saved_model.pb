ΎΛ
ύ
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
Ύ
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
shapeshape"serve*2.2.02unknown8γ
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
Έ
-network_arch/dense_res_block_2/hidden0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-network_arch/dense_res_block_2/hidden0/kernel
±
Anetwork_arch/dense_res_block_2/hidden0/kernel/Read/ReadVariableOpReadVariableOp-network_arch/dense_res_block_2/hidden0/kernel* 
_output_shapes
:
*
dtype0
―
+network_arch/dense_res_block_2/hidden0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+network_arch/dense_res_block_2/hidden0/bias
¨
?network_arch/dense_res_block_2/hidden0/bias/Read/ReadVariableOpReadVariableOp+network_arch/dense_res_block_2/hidden0/bias*
_output_shapes	
:*
dtype0
Έ
-network_arch/dense_res_block_2/hidden1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-network_arch/dense_res_block_2/hidden1/kernel
±
Anetwork_arch/dense_res_block_2/hidden1/kernel/Read/ReadVariableOpReadVariableOp-network_arch/dense_res_block_2/hidden1/kernel* 
_output_shapes
:
*
dtype0
―
+network_arch/dense_res_block_2/hidden1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+network_arch/dense_res_block_2/hidden1/bias
¨
?network_arch/dense_res_block_2/hidden1/bias/Read/ReadVariableOpReadVariableOp+network_arch/dense_res_block_2/hidden1/bias*
_output_shapes	
:*
dtype0
Έ
-network_arch/dense_res_block_2/hidden2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-network_arch/dense_res_block_2/hidden2/kernel
±
Anetwork_arch/dense_res_block_2/hidden2/kernel/Read/ReadVariableOpReadVariableOp-network_arch/dense_res_block_2/hidden2/kernel* 
_output_shapes
:
*
dtype0
―
+network_arch/dense_res_block_2/hidden2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+network_arch/dense_res_block_2/hidden2/bias
¨
?network_arch/dense_res_block_2/hidden2/bias/Read/ReadVariableOpReadVariableOp+network_arch/dense_res_block_2/hidden2/bias*
_output_shapes	
:*
dtype0
Έ
-network_arch/dense_res_block_2/hidden3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-network_arch/dense_res_block_2/hidden3/kernel
±
Anetwork_arch/dense_res_block_2/hidden3/kernel/Read/ReadVariableOpReadVariableOp-network_arch/dense_res_block_2/hidden3/kernel* 
_output_shapes
:
*
dtype0
―
+network_arch/dense_res_block_2/hidden3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+network_arch/dense_res_block_2/hidden3/bias
¨
?network_arch/dense_res_block_2/hidden3/bias/Read/ReadVariableOpReadVariableOp+network_arch/dense_res_block_2/hidden3/bias*
_output_shapes	
:*
dtype0
Ά
,network_arch/dense_res_block_2/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,network_arch/dense_res_block_2/output/kernel
―
@network_arch/dense_res_block_2/output/kernel/Read/ReadVariableOpReadVariableOp,network_arch/dense_res_block_2/output/kernel* 
_output_shapes
:
*
dtype0
­
*network_arch/dense_res_block_2/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*network_arch/dense_res_block_2/output/bias
¦
>network_arch/dense_res_block_2/output/bias/Read/ReadVariableOpReadVariableOp*network_arch/dense_res_block_2/output/bias*
_output_shapes	
:*
dtype0
Έ
-network_arch/dense_res_block_3/hidden0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-network_arch/dense_res_block_3/hidden0/kernel
±
Anetwork_arch/dense_res_block_3/hidden0/kernel/Read/ReadVariableOpReadVariableOp-network_arch/dense_res_block_3/hidden0/kernel* 
_output_shapes
:
*
dtype0
―
+network_arch/dense_res_block_3/hidden0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+network_arch/dense_res_block_3/hidden0/bias
¨
?network_arch/dense_res_block_3/hidden0/bias/Read/ReadVariableOpReadVariableOp+network_arch/dense_res_block_3/hidden0/bias*
_output_shapes	
:*
dtype0
Έ
-network_arch/dense_res_block_3/hidden1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-network_arch/dense_res_block_3/hidden1/kernel
±
Anetwork_arch/dense_res_block_3/hidden1/kernel/Read/ReadVariableOpReadVariableOp-network_arch/dense_res_block_3/hidden1/kernel* 
_output_shapes
:
*
dtype0
―
+network_arch/dense_res_block_3/hidden1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+network_arch/dense_res_block_3/hidden1/bias
¨
?network_arch/dense_res_block_3/hidden1/bias/Read/ReadVariableOpReadVariableOp+network_arch/dense_res_block_3/hidden1/bias*
_output_shapes	
:*
dtype0
Έ
-network_arch/dense_res_block_3/hidden2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-network_arch/dense_res_block_3/hidden2/kernel
±
Anetwork_arch/dense_res_block_3/hidden2/kernel/Read/ReadVariableOpReadVariableOp-network_arch/dense_res_block_3/hidden2/kernel* 
_output_shapes
:
*
dtype0
―
+network_arch/dense_res_block_3/hidden2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+network_arch/dense_res_block_3/hidden2/bias
¨
?network_arch/dense_res_block_3/hidden2/bias/Read/ReadVariableOpReadVariableOp+network_arch/dense_res_block_3/hidden2/bias*
_output_shapes	
:*
dtype0
Έ
-network_arch/dense_res_block_3/hidden3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*>
shared_name/-network_arch/dense_res_block_3/hidden3/kernel
±
Anetwork_arch/dense_res_block_3/hidden3/kernel/Read/ReadVariableOpReadVariableOp-network_arch/dense_res_block_3/hidden3/kernel* 
_output_shapes
:
*
dtype0
―
+network_arch/dense_res_block_3/hidden3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+network_arch/dense_res_block_3/hidden3/bias
¨
?network_arch/dense_res_block_3/hidden3/bias/Read/ReadVariableOpReadVariableOp+network_arch/dense_res_block_3/hidden3/bias*
_output_shapes	
:*
dtype0
Ά
,network_arch/dense_res_block_3/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,network_arch/dense_res_block_3/output/kernel
―
@network_arch/dense_res_block_3/output/kernel/Read/ReadVariableOpReadVariableOp,network_arch/dense_res_block_3/output/kernel* 
_output_shapes
:
*
dtype0
­
*network_arch/dense_res_block_3/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*network_arch/dense_res_block_3/output/bias
¦
>network_arch/dense_res_block_3/output/bias/Read/ReadVariableOpReadVariableOp*network_arch/dense_res_block_3/output/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Μ?
valueΒ?BΏ? BΈ?
ζ
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

	variables
regularization_losses
trainable_variables
	keras_api
^

layers
	variables
regularization_losses
trainable_variables
	keras_api
^

layers
	variables
regularization_losses
trainable_variables
	keras_api
^

kernel
	variables
regularization_losses
trainable_variables
	keras_api
:8
VARIABLE_VALUEVariableL/.ATTRIBUTES/VARIABLE_VALUE
^

kernel
	variables
regularization_losses
 trainable_variables
!	keras_api
 
 
 
 
?
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
20
21
22
 
?
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
20
21
22
­
6metrics

	variables
regularization_losses
7layer_regularization_losses
8non_trainable_variables

9layers
:layer_metrics
trainable_variables
#
;0
<1
=2
>3
?4
F
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
 
F
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
­
@metrics
	variables
regularization_losses
Alayer_regularization_losses
Bnon_trainable_variables

Clayers
Dlayer_metrics
trainable_variables
#
E0
F1
G2
H3
I4
F
,0
-1
.2
/3
04
15
26
37
48
59
 
F
,0
-1
.2
/3
04
15
26
37
48
59
­
Jmetrics
	variables
regularization_losses
Klayer_regularization_losses
Lnon_trainable_variables

Mlayers
Nlayer_metrics
trainable_variables
fd
VARIABLE_VALUE!network_arch/inner_encoder/kernel/inner_encoder/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
­
Ometrics
	variables
regularization_losses
Player_regularization_losses
Qnon_trainable_variables

Rlayers
Slayer_metrics
trainable_variables
fd
VARIABLE_VALUE!network_arch/inner_decoder/kernel/inner_decoder/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
­
Tmetrics
	variables
regularization_losses
Ulayer_regularization_losses
Vnon_trainable_variables

Wlayers
Xlayer_metrics
 trainable_variables
ig
VARIABLE_VALUE-network_arch/dense_res_block_2/hidden0/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+network_arch/dense_res_block_2/hidden0/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE-network_arch/dense_res_block_2/hidden1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+network_arch/dense_res_block_2/hidden1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE-network_arch/dense_res_block_2/hidden2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+network_arch/dense_res_block_2/hidden2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE-network_arch/dense_res_block_2/hidden3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+network_arch/dense_res_block_2/hidden3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE,network_arch/dense_res_block_2/output/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE*network_arch/dense_res_block_2/output/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network_arch/dense_res_block_3/hidden0/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+network_arch/dense_res_block_3/hidden0/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network_arch/dense_res_block_3/hidden1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+network_arch/dense_res_block_3/hidden1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network_arch/dense_res_block_3/hidden2/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+network_arch/dense_res_block_3/hidden2/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-network_arch/dense_res_block_3/hidden3/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+network_arch/dense_res_block_3/hidden3/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,network_arch/dense_res_block_3/output/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*network_arch/dense_res_block_3/output/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
2
3
 
h

"kernel
#bias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
h

$kernel
%bias
]	variables
^regularization_losses
_trainable_variables
`	keras_api
h

&kernel
'bias
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
h

(kernel
)bias
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
h

*kernel
+bias
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
 
 
 
#
;0
<1
=2
>3
?4
 
h

,kernel
-bias
m	variables
nregularization_losses
otrainable_variables
p	keras_api
h

.kernel
/bias
q	variables
rregularization_losses
strainable_variables
t	keras_api
h

0kernel
1bias
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
h

2kernel
3bias
y	variables
zregularization_losses
{trainable_variables
|	keras_api
i

4kernel
5bias
}	variables
~regularization_losses
trainable_variables
	keras_api
 
 
 
#
E0
F1
G2
H3
I4
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

"0
#1
 

"0
#1
²
metrics
Y	variables
Zregularization_losses
 layer_regularization_losses
non_trainable_variables
layers
layer_metrics
[trainable_variables

$0
%1
 

$0
%1
²
metrics
]	variables
^regularization_losses
 layer_regularization_losses
non_trainable_variables
layers
layer_metrics
_trainable_variables

&0
'1
 

&0
'1
²
metrics
a	variables
bregularization_losses
 layer_regularization_losses
non_trainable_variables
layers
layer_metrics
ctrainable_variables

(0
)1
 

(0
)1
²
metrics
e	variables
fregularization_losses
 layer_regularization_losses
non_trainable_variables
layers
layer_metrics
gtrainable_variables

*0
+1
 

*0
+1
²
metrics
i	variables
jregularization_losses
 layer_regularization_losses
non_trainable_variables
layers
layer_metrics
ktrainable_variables

,0
-1
 

,0
-1
²
metrics
m	variables
nregularization_losses
 layer_regularization_losses
non_trainable_variables
layers
layer_metrics
otrainable_variables

.0
/1
 

.0
/1
²
metrics
q	variables
rregularization_losses
  layer_regularization_losses
‘non_trainable_variables
’layers
£layer_metrics
strainable_variables

00
11
 

00
11
²
€metrics
u	variables
vregularization_losses
 ₯layer_regularization_losses
¦non_trainable_variables
§layers
¨layer_metrics
wtrainable_variables

20
31
 

20
31
²
©metrics
y	variables
zregularization_losses
 ͺlayer_regularization_losses
«non_trainable_variables
¬layers
­layer_metrics
{trainable_variables

40
51
 

40
51
²
?metrics
}	variables
~regularization_losses
 ―layer_regularization_losses
°non_trainable_variables
±layers
²layer_metrics
trainable_variables
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
:?????????3*
dtype0*!
shape:?????????3
Ϋ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1-network_arch/dense_res_block_2/hidden0/kernel+network_arch/dense_res_block_2/hidden0/bias-network_arch/dense_res_block_2/hidden1/kernel+network_arch/dense_res_block_2/hidden1/bias-network_arch/dense_res_block_2/hidden2/kernel+network_arch/dense_res_block_2/hidden2/bias-network_arch/dense_res_block_2/hidden3/kernel+network_arch/dense_res_block_2/hidden3/bias,network_arch/dense_res_block_2/output/kernel*network_arch/dense_res_block_2/output/bias!network_arch/inner_encoder/kernel!network_arch/inner_decoder/kernel-network_arch/dense_res_block_3/hidden0/kernel+network_arch/dense_res_block_3/hidden0/bias-network_arch/dense_res_block_3/hidden1/kernel+network_arch/dense_res_block_3/hidden1/bias-network_arch/dense_res_block_3/hidden2/kernel+network_arch/dense_res_block_3/hidden2/bias-network_arch/dense_res_block_3/hidden3/kernel+network_arch/dense_res_block_3/hidden3/bias,network_arch/dense_res_block_3/output/kernel*network_arch/dense_res_block_3/output/biasVariable*#
Tin
2*
Tout
2*\
_output_shapesJ
H:?????????3:?????????3:?????????2*9
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*/
f*R(
&__inference_signature_wrapper_16862664
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Β
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp5network_arch/inner_encoder/kernel/Read/ReadVariableOp5network_arch/inner_decoder/kernel/Read/ReadVariableOpAnetwork_arch/dense_res_block_2/hidden0/kernel/Read/ReadVariableOp?network_arch/dense_res_block_2/hidden0/bias/Read/ReadVariableOpAnetwork_arch/dense_res_block_2/hidden1/kernel/Read/ReadVariableOp?network_arch/dense_res_block_2/hidden1/bias/Read/ReadVariableOpAnetwork_arch/dense_res_block_2/hidden2/kernel/Read/ReadVariableOp?network_arch/dense_res_block_2/hidden2/bias/Read/ReadVariableOpAnetwork_arch/dense_res_block_2/hidden3/kernel/Read/ReadVariableOp?network_arch/dense_res_block_2/hidden3/bias/Read/ReadVariableOp@network_arch/dense_res_block_2/output/kernel/Read/ReadVariableOp>network_arch/dense_res_block_2/output/bias/Read/ReadVariableOpAnetwork_arch/dense_res_block_3/hidden0/kernel/Read/ReadVariableOp?network_arch/dense_res_block_3/hidden0/bias/Read/ReadVariableOpAnetwork_arch/dense_res_block_3/hidden1/kernel/Read/ReadVariableOp?network_arch/dense_res_block_3/hidden1/bias/Read/ReadVariableOpAnetwork_arch/dense_res_block_3/hidden2/kernel/Read/ReadVariableOp?network_arch/dense_res_block_3/hidden2/bias/Read/ReadVariableOpAnetwork_arch/dense_res_block_3/hidden3/kernel/Read/ReadVariableOp?network_arch/dense_res_block_3/hidden3/bias/Read/ReadVariableOp@network_arch/dense_res_block_3/output/kernel/Read/ReadVariableOp>network_arch/dense_res_block_3/output/bias/Read/ReadVariableOpConst*$
Tin
2*
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
!__inference__traced_save_16863054
ρ	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable!network_arch/inner_encoder/kernel!network_arch/inner_decoder/kernel-network_arch/dense_res_block_2/hidden0/kernel+network_arch/dense_res_block_2/hidden0/bias-network_arch/dense_res_block_2/hidden1/kernel+network_arch/dense_res_block_2/hidden1/bias-network_arch/dense_res_block_2/hidden2/kernel+network_arch/dense_res_block_2/hidden2/bias-network_arch/dense_res_block_2/hidden3/kernel+network_arch/dense_res_block_2/hidden3/bias,network_arch/dense_res_block_2/output/kernel*network_arch/dense_res_block_2/output/bias-network_arch/dense_res_block_3/hidden0/kernel+network_arch/dense_res_block_3/hidden0/bias-network_arch/dense_res_block_3/hidden1/kernel+network_arch/dense_res_block_3/hidden1/bias-network_arch/dense_res_block_3/hidden2/kernel+network_arch/dense_res_block_3/hidden2/bias-network_arch/dense_res_block_3/hidden3/kernel+network_arch/dense_res_block_3/hidden3/bias,network_arch/dense_res_block_3/output/kernel*network_arch/dense_res_block_3/output/bias*#
Tin
2*
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
$__inference__traced_restore_16863135υ
Θ
v
0__inference_inner_encoder_layer_call_fn_16860965

inputs
unknown
identity’StatefulPartitionedCallΜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*'
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_encoder_layer_call_and_return_conditional_losses_168574512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
ΰ
h
__inference_loss_fn_6_168628115
1kernel_regularizer_square_readvariableop_resource
identityΘ
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
 *wΜ+22
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
ς
Ό-
J__inference_network_arch_layer_call_and_return_conditional_losses_16860831
input_1
dense_res_block_2_5929713
dense_res_block_2_5929715
dense_res_block_2_5929717
dense_res_block_2_5929719
dense_res_block_2_5929721
dense_res_block_2_5929723
dense_res_block_2_5929725
dense_res_block_2_5929727
dense_res_block_2_5929729
dense_res_block_2_5929731
inner_encoder_5929759
inner_decoder_5929787
dense_res_block_3_5929902
dense_res_block_3_5929904
dense_res_block_3_5929906
dense_res_block_3_5929908
dense_res_block_3_5929910
dense_res_block_3_5929912
dense_res_block_3_5929914
dense_res_block_3_5929916
dense_res_block_3_5929918
dense_res_block_3_5929920%
!diag_part_readvariableop_resource
identity_100
identity_101
identity_102’)dense_res_block_2/StatefulPartitionedCall’+dense_res_block_2_1/StatefulPartitionedCall’+dense_res_block_2_2/StatefulPartitionedCall’+dense_res_block_2_3/StatefulPartitionedCall’)dense_res_block_3/StatefulPartitionedCall’+dense_res_block_3_1/StatefulPartitionedCall’,dense_res_block_3_10/StatefulPartitionedCall’,dense_res_block_3_11/StatefulPartitionedCall’,dense_res_block_3_12/StatefulPartitionedCall’,dense_res_block_3_13/StatefulPartitionedCall’,dense_res_block_3_14/StatefulPartitionedCall’,dense_res_block_3_15/StatefulPartitionedCall’,dense_res_block_3_16/StatefulPartitionedCall’,dense_res_block_3_17/StatefulPartitionedCall’,dense_res_block_3_18/StatefulPartitionedCall’,dense_res_block_3_19/StatefulPartitionedCall’+dense_res_block_3_2/StatefulPartitionedCall’,dense_res_block_3_20/StatefulPartitionedCall’,dense_res_block_3_21/StatefulPartitionedCall’,dense_res_block_3_22/StatefulPartitionedCall’,dense_res_block_3_23/StatefulPartitionedCall’,dense_res_block_3_24/StatefulPartitionedCall’,dense_res_block_3_25/StatefulPartitionedCall’,dense_res_block_3_26/StatefulPartitionedCall’,dense_res_block_3_27/StatefulPartitionedCall’,dense_res_block_3_28/StatefulPartitionedCall’,dense_res_block_3_29/StatefulPartitionedCall’+dense_res_block_3_3/StatefulPartitionedCall’,dense_res_block_3_30/StatefulPartitionedCall’,dense_res_block_3_31/StatefulPartitionedCall’,dense_res_block_3_32/StatefulPartitionedCall’,dense_res_block_3_33/StatefulPartitionedCall’,dense_res_block_3_34/StatefulPartitionedCall’,dense_res_block_3_35/StatefulPartitionedCall’,dense_res_block_3_36/StatefulPartitionedCall’,dense_res_block_3_37/StatefulPartitionedCall’,dense_res_block_3_38/StatefulPartitionedCall’,dense_res_block_3_39/StatefulPartitionedCall’+dense_res_block_3_4/StatefulPartitionedCall’,dense_res_block_3_40/StatefulPartitionedCall’,dense_res_block_3_41/StatefulPartitionedCall’,dense_res_block_3_42/StatefulPartitionedCall’,dense_res_block_3_43/StatefulPartitionedCall’,dense_res_block_3_44/StatefulPartitionedCall’,dense_res_block_3_45/StatefulPartitionedCall’,dense_res_block_3_46/StatefulPartitionedCall’,dense_res_block_3_47/StatefulPartitionedCall’,dense_res_block_3_48/StatefulPartitionedCall’,dense_res_block_3_49/StatefulPartitionedCall’+dense_res_block_3_5/StatefulPartitionedCall’,dense_res_block_3_50/StatefulPartitionedCall’,dense_res_block_3_51/StatefulPartitionedCall’+dense_res_block_3_6/StatefulPartitionedCall’+dense_res_block_3_7/StatefulPartitionedCall’+dense_res_block_3_8/StatefulPartitionedCall’+dense_res_block_3_9/StatefulPartitionedCall’%inner_decoder/StatefulPartitionedCall’'inner_decoder_1/StatefulPartitionedCall’(inner_decoder_10/StatefulPartitionedCall’(inner_decoder_11/StatefulPartitionedCall’(inner_decoder_12/StatefulPartitionedCall’(inner_decoder_13/StatefulPartitionedCall’(inner_decoder_14/StatefulPartitionedCall’(inner_decoder_15/StatefulPartitionedCall’(inner_decoder_16/StatefulPartitionedCall’(inner_decoder_17/StatefulPartitionedCall’(inner_decoder_18/StatefulPartitionedCall’(inner_decoder_19/StatefulPartitionedCall’'inner_decoder_2/StatefulPartitionedCall’(inner_decoder_20/StatefulPartitionedCall’(inner_decoder_21/StatefulPartitionedCall’(inner_decoder_22/StatefulPartitionedCall’(inner_decoder_23/StatefulPartitionedCall’(inner_decoder_24/StatefulPartitionedCall’(inner_decoder_25/StatefulPartitionedCall’(inner_decoder_26/StatefulPartitionedCall’(inner_decoder_27/StatefulPartitionedCall’(inner_decoder_28/StatefulPartitionedCall’(inner_decoder_29/StatefulPartitionedCall’'inner_decoder_3/StatefulPartitionedCall’(inner_decoder_30/StatefulPartitionedCall’(inner_decoder_31/StatefulPartitionedCall’(inner_decoder_32/StatefulPartitionedCall’(inner_decoder_33/StatefulPartitionedCall’(inner_decoder_34/StatefulPartitionedCall’(inner_decoder_35/StatefulPartitionedCall’(inner_decoder_36/StatefulPartitionedCall’(inner_decoder_37/StatefulPartitionedCall’(inner_decoder_38/StatefulPartitionedCall’(inner_decoder_39/StatefulPartitionedCall’'inner_decoder_4/StatefulPartitionedCall’(inner_decoder_40/StatefulPartitionedCall’(inner_decoder_41/StatefulPartitionedCall’(inner_decoder_42/StatefulPartitionedCall’(inner_decoder_43/StatefulPartitionedCall’(inner_decoder_44/StatefulPartitionedCall’(inner_decoder_45/StatefulPartitionedCall’(inner_decoder_46/StatefulPartitionedCall’(inner_decoder_47/StatefulPartitionedCall’(inner_decoder_48/StatefulPartitionedCall’(inner_decoder_49/StatefulPartitionedCall’'inner_decoder_5/StatefulPartitionedCall’(inner_decoder_50/StatefulPartitionedCall’'inner_decoder_6/StatefulPartitionedCall’'inner_decoder_7/StatefulPartitionedCall’'inner_decoder_8/StatefulPartitionedCall’'inner_decoder_9/StatefulPartitionedCall’%inner_encoder/StatefulPartitionedCall’'inner_encoder_1/StatefulPartitionedCall’'inner_encoder_2/StatefulPartitionedCall’'inner_encoder_3/StatefulPartitionedCall
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
strided_slice/stack_2ϋ
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????22
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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

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
:?????????*

begin_mask*
end_mask2
strided_slice_101`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axisΐ
concat_1ConcatV2strided_slice_52:output:0strided_slice_53:output:0strided_slice_54:output:0strided_slice_55:output:0strided_slice_56:output:0strided_slice_57:output:0strided_slice_58:output:0strided_slice_59:output:0strided_slice_60:output:0strided_slice_61:output:0strided_slice_62:output:0strided_slice_63:output:0strided_slice_64:output:0strided_slice_65:output:0strided_slice_66:output:0strided_slice_67:output:0strided_slice_68:output:0strided_slice_69:output:0strided_slice_70:output:0strided_slice_71:output:0strided_slice_72:output:0strided_slice_73:output:0strided_slice_74:output:0strided_slice_75:output:0strided_slice_76:output:0strided_slice_77:output:0strided_slice_78:output:0strided_slice_79:output:0strided_slice_80:output:0strided_slice_81:output:0strided_slice_82:output:0strided_slice_83:output:0strided_slice_84:output:0strided_slice_85:output:0strided_slice_86:output:0strided_slice_87:output:0strided_slice_88:output:0strided_slice_89:output:0strided_slice_90:output:0strided_slice_91:output:0strided_slice_92:output:0strided_slice_93:output:0strided_slice_94:output:0strided_slice_95:output:0strided_slice_96:output:0strided_slice_97:output:0strided_slice_98:output:0strided_slice_99:output:0strided_slice_100:output:0strided_slice_101:output:0concat_1/axis:output:0*
N2*
T0*,
_output_shapes
:?????????22

concat_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapeq
ReshapeReshapeinput_1Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2	
Reshapes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_1/shape
	Reshape_1Reshapestrided_slice:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:?????????2
	Reshape_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_2/shape
	Reshape_2Reshapestrided_slice_1:output:0Reshape_2/shape:output:0*
T0*(
_output_shapes
:?????????2
	Reshape_2s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_3/shape
	Reshape_3Reshapeconcat_1:output:0Reshape_3/shape:output:0*
T0*(
_output_shapes
:?????????2
	Reshape_3
)dense_res_block_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_res_block_2_5929713dense_res_block_2_5929715dense_res_block_2_5929717dense_res_block_2_5929719dense_res_block_2_5929721dense_res_block_2_5929723dense_res_block_2_5929725dense_res_block_2_5929727dense_res_block_2_5929729dense_res_block_2_5929731*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_2_layer_call_and_return_conditional_losses_168567832+
)dense_res_block_2/StatefulPartitionedCallΧ
dense_res_block_2/IdentityIdentity2dense_res_block_2/StatefulPartitionedCall:output:0*^dense_res_block_2/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_2/Identity
%inner_encoder/StatefulPartitionedCallStatefulPartitionedCall#dense_res_block_2/Identity:output:0inner_encoder_5929759*
Tin
2*
Tout
2*'
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_encoder_layer_call_and_return_conditional_losses_168574512'
%inner_encoder/StatefulPartitionedCallΖ
inner_encoder/IdentityIdentity.inner_encoder/StatefulPartitionedCall:output:0&^inner_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
inner_encoder/Identity
%inner_decoder/StatefulPartitionedCallStatefulPartitionedCallinner_encoder/Identity:output:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132'
%inner_decoder/StatefulPartitionedCallΗ
inner_decoder/IdentityIdentity.inner_decoder/StatefulPartitionedCall:output:0&^inner_decoder/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder/Identity₯
)dense_res_block_3/StatefulPartitionedCallStatefulPartitionedCallinner_decoder/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332+
)dense_res_block_3/StatefulPartitionedCallΧ
dense_res_block_3/IdentityIdentity2dense_res_block_3/StatefulPartitionedCall:output:0*^dense_res_block_3/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3/Identityw
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????3      2
Reshape_4/shape
	Reshape_4Reshape#dense_res_block_3/Identity:output:0Reshape_4/shape:output:0*
T0*,
_output_shapes
:?????????32
	Reshape_4­
+dense_res_block_3_1/StatefulPartitionedCallStatefulPartitionedCall#dense_res_block_2/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332-
+dense_res_block_3_1/StatefulPartitionedCallί
dense_res_block_3_1/IdentityIdentity4dense_res_block_3_1/StatefulPartitionedCall:output:0,^dense_res_block_3_1/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_1/Identityw
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????3      2
Reshape_5/shape
	Reshape_5Reshape%dense_res_block_3_1/Identity:output:0Reshape_5/shape:output:0*
T0*,
_output_shapes
:?????????32
	Reshape_5

RelMSE/subSubinner_decoder/Identity:output:0#dense_res_block_2/Identity:output:0*
T0*(
_output_shapes
:?????????2

RelMSE/subk
RelMSE/SquareSquareRelMSE/sub:z:0*
T0*(
_output_shapes
:?????????2
RelMSE/Square
RelMSE/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
RelMSE/Mean/reduction_indices
RelMSE/MeanMeanRelMSE/Square:y:0&RelMSE/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
RelMSE/Mean
RelMSE/Square_1Square#dense_res_block_2/Identity:output:0*
T0*(
_output_shapes
:?????????2
RelMSE/Square_1
RelMSE/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
RelMSE/Mean_1/reduction_indices
RelMSE/Mean_1MeanRelMSE/Square_1:y:0(RelMSE/Mean_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
RelMSE/Mean_1a
RelMSE/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'72
RelMSE/add/y~

RelMSE/addAddV2RelMSE/Mean_1:output:0RelMSE/add/y:output:0*
T0*#
_output_shapes
:?????????2

RelMSE/add
RelMSE/truedivRealDivRelMSE/Mean:output:0RelMSE/add:z:0*
T0*#
_output_shapes
:?????????2
RelMSE/truediv
RelMSE/Mean_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
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
!RelMSE/weighted_loss/num_elements΄
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
RelMSE/weighted_loss/Const_1ͺ
RelMSE/weighted_loss/Sum_1Sum!RelMSE/weighted_loss/Sum:output:0%RelMSE/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 2
RelMSE/weighted_loss/Sum_1Ά
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
mul
diag_part/ReadVariableOpReadVariableOp!diag_part_readvariableop_resource*
_output_shapes

:*
dtype02
diag_part/ReadVariableOp\
diag_part/kConst*
_output_shapes
: *
dtype0*
value	B : 2
diag_part/kw
diag_part/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
diag_part/padding_value©
	diag_partMatrixDiagPartV3 diag_part/ReadVariableOp:value:0diag_part/k:output:0 diag_part/padding_value:output:0*
T0*
_output_shapes
:2
	diag_partR
diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
diag/ki
diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
diag/num_rowsi
diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
diag/num_colsm
diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
diag/padding_valueΉ
diagMatrixDiagV3diag_part:diagonal:0diag/k:output:0diag/num_rows:output:0diag/num_cols:output:0diag/padding_value:output:0*
T0*
_output_shapes

:2
diag
+dense_res_block_2_1/StatefulPartitionedCallStatefulPartitionedCallReshape_1:output:0dense_res_block_2_5929713dense_res_block_2_5929715dense_res_block_2_5929717dense_res_block_2_5929719dense_res_block_2_5929721dense_res_block_2_5929723dense_res_block_2_5929725dense_res_block_2_5929727dense_res_block_2_5929729dense_res_block_2_5929731*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_2_layer_call_and_return_conditional_losses_168567832-
+dense_res_block_2_1/StatefulPartitionedCallί
dense_res_block_2_1/IdentityIdentity4dense_res_block_2_1/StatefulPartitionedCall:output:0,^dense_res_block_2_1/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_2_1/Identity
'inner_encoder_1/StatefulPartitionedCallStatefulPartitionedCall%dense_res_block_2_1/Identity:output:0inner_encoder_5929759*
Tin
2*
Tout
2*'
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_encoder_layer_call_and_return_conditional_losses_168574512)
'inner_encoder_1/StatefulPartitionedCallΞ
inner_encoder_1/IdentityIdentity0inner_encoder_1/StatefulPartitionedCall:output:0(^inner_encoder_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
inner_encoder_1/Identity~
MatMulMatMul!inner_encoder_1/Identity:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
MatMul
'inner_decoder_1/StatefulPartitionedCallStatefulPartitionedCallMatMul:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132)
'inner_decoder_1/StatefulPartitionedCallΟ
inner_decoder_1/IdentityIdentity0inner_decoder_1/StatefulPartitionedCall:output:0(^inner_decoder_1/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_1/Identity«
+dense_res_block_3_2/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_1/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332-
+dense_res_block_3_2/StatefulPartitionedCallί
dense_res_block_3_2/IdentityIdentity4dense_res_block_3_2/StatefulPartitionedCall:output:0,^dense_res_block_3_2/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_2/Identityw
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_6/shape
	Reshape_6Reshape%dense_res_block_3_2/Identity:output:0Reshape_6/shape:output:0*
T0*,
_output_shapes
:?????????2
	Reshape_6d
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:?????????2

Identityr
MatMul_1MatMulIdentity:output:0diag:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1
'inner_decoder_2/StatefulPartitionedCallStatefulPartitionedCallMatMul_1:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132)
'inner_decoder_2/StatefulPartitionedCallΟ
inner_decoder_2/IdentityIdentity0inner_decoder_2/StatefulPartitionedCall:output:0(^inner_decoder_2/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_2/Identity«
+dense_res_block_3_3/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_2/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332-
+dense_res_block_3_3/StatefulPartitionedCallί
dense_res_block_3_3/IdentityIdentity4dense_res_block_3_3/StatefulPartitionedCall:output:0,^dense_res_block_3_3/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_3/Identityw
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_7/shape
	Reshape_7Reshape%dense_res_block_3_3/Identity:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:?????????2
	Reshape_7j

Identity_1IdentityMatMul_1:product:0*
T0*'
_output_shapes
:?????????2

Identity_1t
MatMul_2MatMulIdentity_1:output:0diag:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2
'inner_decoder_3/StatefulPartitionedCallStatefulPartitionedCallMatMul_2:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132)
'inner_decoder_3/StatefulPartitionedCallΟ
inner_decoder_3/IdentityIdentity0inner_decoder_3/StatefulPartitionedCall:output:0(^inner_decoder_3/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_3/Identity«
+dense_res_block_3_4/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_3/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332-
+dense_res_block_3_4/StatefulPartitionedCallί
dense_res_block_3_4/IdentityIdentity4dense_res_block_3_4/StatefulPartitionedCall:output:0,^dense_res_block_3_4/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_4/Identityw
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_8/shape
	Reshape_8Reshape%dense_res_block_3_4/Identity:output:0Reshape_8/shape:output:0*
T0*,
_output_shapes
:?????????2
	Reshape_8j

Identity_2IdentityMatMul_2:product:0*
T0*'
_output_shapes
:?????????2

Identity_2t
MatMul_3MatMulIdentity_2:output:0diag:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3
'inner_decoder_4/StatefulPartitionedCallStatefulPartitionedCallMatMul_3:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132)
'inner_decoder_4/StatefulPartitionedCallΟ
inner_decoder_4/IdentityIdentity0inner_decoder_4/StatefulPartitionedCall:output:0(^inner_decoder_4/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_4/Identity«
+dense_res_block_3_5/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_4/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332-
+dense_res_block_3_5/StatefulPartitionedCallί
dense_res_block_3_5/IdentityIdentity4dense_res_block_3_5/StatefulPartitionedCall:output:0,^dense_res_block_3_5/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_5/Identityw
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_9/shape
	Reshape_9Reshape%dense_res_block_3_5/Identity:output:0Reshape_9/shape:output:0*
T0*,
_output_shapes
:?????????2
	Reshape_9j

Identity_3IdentityMatMul_3:product:0*
T0*'
_output_shapes
:?????????2

Identity_3t
MatMul_4MatMulIdentity_3:output:0diag:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4
'inner_decoder_5/StatefulPartitionedCallStatefulPartitionedCallMatMul_4:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132)
'inner_decoder_5/StatefulPartitionedCallΟ
inner_decoder_5/IdentityIdentity0inner_decoder_5/StatefulPartitionedCall:output:0(^inner_decoder_5/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_5/Identity«
+dense_res_block_3_6/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_5/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332-
+dense_res_block_3_6/StatefulPartitionedCallί
dense_res_block_3_6/IdentityIdentity4dense_res_block_3_6/StatefulPartitionedCall:output:0,^dense_res_block_3_6/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_6/Identityy
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_10/shape

Reshape_10Reshape%dense_res_block_3_6/Identity:output:0Reshape_10/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_10j

Identity_4IdentityMatMul_4:product:0*
T0*'
_output_shapes
:?????????2

Identity_4t
MatMul_5MatMulIdentity_4:output:0diag:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5
'inner_decoder_6/StatefulPartitionedCallStatefulPartitionedCallMatMul_5:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132)
'inner_decoder_6/StatefulPartitionedCallΟ
inner_decoder_6/IdentityIdentity0inner_decoder_6/StatefulPartitionedCall:output:0(^inner_decoder_6/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_6/Identity«
+dense_res_block_3_7/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_6/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332-
+dense_res_block_3_7/StatefulPartitionedCallί
dense_res_block_3_7/IdentityIdentity4dense_res_block_3_7/StatefulPartitionedCall:output:0,^dense_res_block_3_7/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_7/Identityy
Reshape_11/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_11/shape

Reshape_11Reshape%dense_res_block_3_7/Identity:output:0Reshape_11/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_11j

Identity_5IdentityMatMul_5:product:0*
T0*'
_output_shapes
:?????????2

Identity_5t
MatMul_6MatMulIdentity_5:output:0diag:output:0*
T0*'
_output_shapes
:?????????2

MatMul_6
'inner_decoder_7/StatefulPartitionedCallStatefulPartitionedCallMatMul_6:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132)
'inner_decoder_7/StatefulPartitionedCallΟ
inner_decoder_7/IdentityIdentity0inner_decoder_7/StatefulPartitionedCall:output:0(^inner_decoder_7/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_7/Identity«
+dense_res_block_3_8/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_7/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332-
+dense_res_block_3_8/StatefulPartitionedCallί
dense_res_block_3_8/IdentityIdentity4dense_res_block_3_8/StatefulPartitionedCall:output:0,^dense_res_block_3_8/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_8/Identityy
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_12/shape

Reshape_12Reshape%dense_res_block_3_8/Identity:output:0Reshape_12/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_12j

Identity_6IdentityMatMul_6:product:0*
T0*'
_output_shapes
:?????????2

Identity_6t
MatMul_7MatMulIdentity_6:output:0diag:output:0*
T0*'
_output_shapes
:?????????2

MatMul_7
'inner_decoder_8/StatefulPartitionedCallStatefulPartitionedCallMatMul_7:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132)
'inner_decoder_8/StatefulPartitionedCallΟ
inner_decoder_8/IdentityIdentity0inner_decoder_8/StatefulPartitionedCall:output:0(^inner_decoder_8/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_8/Identity«
+dense_res_block_3_9/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_8/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332-
+dense_res_block_3_9/StatefulPartitionedCallί
dense_res_block_3_9/IdentityIdentity4dense_res_block_3_9/StatefulPartitionedCall:output:0,^dense_res_block_3_9/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_9/Identityy
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_13/shape

Reshape_13Reshape%dense_res_block_3_9/Identity:output:0Reshape_13/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_13j

Identity_7IdentityMatMul_7:product:0*
T0*'
_output_shapes
:?????????2

Identity_7t
MatMul_8MatMulIdentity_7:output:0diag:output:0*
T0*'
_output_shapes
:?????????2

MatMul_8
'inner_decoder_9/StatefulPartitionedCallStatefulPartitionedCallMatMul_8:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132)
'inner_decoder_9/StatefulPartitionedCallΟ
inner_decoder_9/IdentityIdentity0inner_decoder_9/StatefulPartitionedCall:output:0(^inner_decoder_9/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_9/Identity­
,dense_res_block_3_10/StatefulPartitionedCallStatefulPartitionedCall!inner_decoder_9/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_10/StatefulPartitionedCallγ
dense_res_block_3_10/IdentityIdentity5dense_res_block_3_10/StatefulPartitionedCall:output:0-^dense_res_block_3_10/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_10/Identityy
Reshape_14/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_14/shape

Reshape_14Reshape&dense_res_block_3_10/Identity:output:0Reshape_14/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_14j

Identity_8IdentityMatMul_8:product:0*
T0*'
_output_shapes
:?????????2

Identity_8t
MatMul_9MatMulIdentity_8:output:0diag:output:0*
T0*'
_output_shapes
:?????????2

MatMul_9
(inner_decoder_10/StatefulPartitionedCallStatefulPartitionedCallMatMul_9:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_10/StatefulPartitionedCallΣ
inner_decoder_10/IdentityIdentity1inner_decoder_10/StatefulPartitionedCall:output:0)^inner_decoder_10/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_10/Identity?
,dense_res_block_3_11/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_10/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_11/StatefulPartitionedCallγ
dense_res_block_3_11/IdentityIdentity5dense_res_block_3_11/StatefulPartitionedCall:output:0-^dense_res_block_3_11/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_11/Identityy
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_15/shape

Reshape_15Reshape&dense_res_block_3_11/Identity:output:0Reshape_15/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_15j

Identity_9IdentityMatMul_9:product:0*
T0*'
_output_shapes
:?????????2

Identity_9v
	MatMul_10MatMulIdentity_9:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_10
(inner_decoder_11/StatefulPartitionedCallStatefulPartitionedCallMatMul_10:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_11/StatefulPartitionedCallΣ
inner_decoder_11/IdentityIdentity1inner_decoder_11/StatefulPartitionedCall:output:0)^inner_decoder_11/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_11/Identity?
,dense_res_block_3_12/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_11/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_12/StatefulPartitionedCallγ
dense_res_block_3_12/IdentityIdentity5dense_res_block_3_12/StatefulPartitionedCall:output:0-^dense_res_block_3_12/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_12/Identityy
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_16/shape

Reshape_16Reshape&dense_res_block_3_12/Identity:output:0Reshape_16/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_16m
Identity_10IdentityMatMul_10:product:0*
T0*'
_output_shapes
:?????????2
Identity_10w
	MatMul_11MatMulIdentity_10:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_11
(inner_decoder_12/StatefulPartitionedCallStatefulPartitionedCallMatMul_11:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_12/StatefulPartitionedCallΣ
inner_decoder_12/IdentityIdentity1inner_decoder_12/StatefulPartitionedCall:output:0)^inner_decoder_12/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_12/Identity?
,dense_res_block_3_13/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_12/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_13/StatefulPartitionedCallγ
dense_res_block_3_13/IdentityIdentity5dense_res_block_3_13/StatefulPartitionedCall:output:0-^dense_res_block_3_13/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_13/Identityy
Reshape_17/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_17/shape

Reshape_17Reshape&dense_res_block_3_13/Identity:output:0Reshape_17/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_17m
Identity_11IdentityMatMul_11:product:0*
T0*'
_output_shapes
:?????????2
Identity_11w
	MatMul_12MatMulIdentity_11:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_12
(inner_decoder_13/StatefulPartitionedCallStatefulPartitionedCallMatMul_12:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_13/StatefulPartitionedCallΣ
inner_decoder_13/IdentityIdentity1inner_decoder_13/StatefulPartitionedCall:output:0)^inner_decoder_13/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_13/Identity?
,dense_res_block_3_14/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_13/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_14/StatefulPartitionedCallγ
dense_res_block_3_14/IdentityIdentity5dense_res_block_3_14/StatefulPartitionedCall:output:0-^dense_res_block_3_14/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_14/Identityy
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_18/shape

Reshape_18Reshape&dense_res_block_3_14/Identity:output:0Reshape_18/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_18m
Identity_12IdentityMatMul_12:product:0*
T0*'
_output_shapes
:?????????2
Identity_12w
	MatMul_13MatMulIdentity_12:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_13
(inner_decoder_14/StatefulPartitionedCallStatefulPartitionedCallMatMul_13:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_14/StatefulPartitionedCallΣ
inner_decoder_14/IdentityIdentity1inner_decoder_14/StatefulPartitionedCall:output:0)^inner_decoder_14/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_14/Identity?
,dense_res_block_3_15/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_14/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_15/StatefulPartitionedCallγ
dense_res_block_3_15/IdentityIdentity5dense_res_block_3_15/StatefulPartitionedCall:output:0-^dense_res_block_3_15/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_15/Identityy
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_19/shape

Reshape_19Reshape&dense_res_block_3_15/Identity:output:0Reshape_19/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_19m
Identity_13IdentityMatMul_13:product:0*
T0*'
_output_shapes
:?????????2
Identity_13w
	MatMul_14MatMulIdentity_13:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_14
(inner_decoder_15/StatefulPartitionedCallStatefulPartitionedCallMatMul_14:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_15/StatefulPartitionedCallΣ
inner_decoder_15/IdentityIdentity1inner_decoder_15/StatefulPartitionedCall:output:0)^inner_decoder_15/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_15/Identity?
,dense_res_block_3_16/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_15/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_16/StatefulPartitionedCallγ
dense_res_block_3_16/IdentityIdentity5dense_res_block_3_16/StatefulPartitionedCall:output:0-^dense_res_block_3_16/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_16/Identityy
Reshape_20/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_20/shape

Reshape_20Reshape&dense_res_block_3_16/Identity:output:0Reshape_20/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_20m
Identity_14IdentityMatMul_14:product:0*
T0*'
_output_shapes
:?????????2
Identity_14w
	MatMul_15MatMulIdentity_14:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_15
(inner_decoder_16/StatefulPartitionedCallStatefulPartitionedCallMatMul_15:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_16/StatefulPartitionedCallΣ
inner_decoder_16/IdentityIdentity1inner_decoder_16/StatefulPartitionedCall:output:0)^inner_decoder_16/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_16/Identity?
,dense_res_block_3_17/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_16/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_17/StatefulPartitionedCallγ
dense_res_block_3_17/IdentityIdentity5dense_res_block_3_17/StatefulPartitionedCall:output:0-^dense_res_block_3_17/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_17/Identityy
Reshape_21/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_21/shape

Reshape_21Reshape&dense_res_block_3_17/Identity:output:0Reshape_21/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_21m
Identity_15IdentityMatMul_15:product:0*
T0*'
_output_shapes
:?????????2
Identity_15w
	MatMul_16MatMulIdentity_15:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_16
(inner_decoder_17/StatefulPartitionedCallStatefulPartitionedCallMatMul_16:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_17/StatefulPartitionedCallΣ
inner_decoder_17/IdentityIdentity1inner_decoder_17/StatefulPartitionedCall:output:0)^inner_decoder_17/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_17/Identity?
,dense_res_block_3_18/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_17/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_18/StatefulPartitionedCallγ
dense_res_block_3_18/IdentityIdentity5dense_res_block_3_18/StatefulPartitionedCall:output:0-^dense_res_block_3_18/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_18/Identityy
Reshape_22/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_22/shape

Reshape_22Reshape&dense_res_block_3_18/Identity:output:0Reshape_22/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_22m
Identity_16IdentityMatMul_16:product:0*
T0*'
_output_shapes
:?????????2
Identity_16w
	MatMul_17MatMulIdentity_16:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_17
(inner_decoder_18/StatefulPartitionedCallStatefulPartitionedCallMatMul_17:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_18/StatefulPartitionedCallΣ
inner_decoder_18/IdentityIdentity1inner_decoder_18/StatefulPartitionedCall:output:0)^inner_decoder_18/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_18/Identity?
,dense_res_block_3_19/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_18/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_19/StatefulPartitionedCallγ
dense_res_block_3_19/IdentityIdentity5dense_res_block_3_19/StatefulPartitionedCall:output:0-^dense_res_block_3_19/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_19/Identityy
Reshape_23/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_23/shape

Reshape_23Reshape&dense_res_block_3_19/Identity:output:0Reshape_23/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_23m
Identity_17IdentityMatMul_17:product:0*
T0*'
_output_shapes
:?????????2
Identity_17w
	MatMul_18MatMulIdentity_17:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_18
(inner_decoder_19/StatefulPartitionedCallStatefulPartitionedCallMatMul_18:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_19/StatefulPartitionedCallΣ
inner_decoder_19/IdentityIdentity1inner_decoder_19/StatefulPartitionedCall:output:0)^inner_decoder_19/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_19/Identity?
,dense_res_block_3_20/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_19/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_20/StatefulPartitionedCallγ
dense_res_block_3_20/IdentityIdentity5dense_res_block_3_20/StatefulPartitionedCall:output:0-^dense_res_block_3_20/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_20/Identityy
Reshape_24/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_24/shape

Reshape_24Reshape&dense_res_block_3_20/Identity:output:0Reshape_24/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_24m
Identity_18IdentityMatMul_18:product:0*
T0*'
_output_shapes
:?????????2
Identity_18w
	MatMul_19MatMulIdentity_18:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_19
(inner_decoder_20/StatefulPartitionedCallStatefulPartitionedCallMatMul_19:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_20/StatefulPartitionedCallΣ
inner_decoder_20/IdentityIdentity1inner_decoder_20/StatefulPartitionedCall:output:0)^inner_decoder_20/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_20/Identity?
,dense_res_block_3_21/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_20/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_21/StatefulPartitionedCallγ
dense_res_block_3_21/IdentityIdentity5dense_res_block_3_21/StatefulPartitionedCall:output:0-^dense_res_block_3_21/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_21/Identityy
Reshape_25/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_25/shape

Reshape_25Reshape&dense_res_block_3_21/Identity:output:0Reshape_25/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_25m
Identity_19IdentityMatMul_19:product:0*
T0*'
_output_shapes
:?????????2
Identity_19w
	MatMul_20MatMulIdentity_19:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_20
(inner_decoder_21/StatefulPartitionedCallStatefulPartitionedCallMatMul_20:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_21/StatefulPartitionedCallΣ
inner_decoder_21/IdentityIdentity1inner_decoder_21/StatefulPartitionedCall:output:0)^inner_decoder_21/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_21/Identity?
,dense_res_block_3_22/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_21/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_22/StatefulPartitionedCallγ
dense_res_block_3_22/IdentityIdentity5dense_res_block_3_22/StatefulPartitionedCall:output:0-^dense_res_block_3_22/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_22/Identityy
Reshape_26/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_26/shape

Reshape_26Reshape&dense_res_block_3_22/Identity:output:0Reshape_26/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_26m
Identity_20IdentityMatMul_20:product:0*
T0*'
_output_shapes
:?????????2
Identity_20w
	MatMul_21MatMulIdentity_20:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_21
(inner_decoder_22/StatefulPartitionedCallStatefulPartitionedCallMatMul_21:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_22/StatefulPartitionedCallΣ
inner_decoder_22/IdentityIdentity1inner_decoder_22/StatefulPartitionedCall:output:0)^inner_decoder_22/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_22/Identity?
,dense_res_block_3_23/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_22/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_23/StatefulPartitionedCallγ
dense_res_block_3_23/IdentityIdentity5dense_res_block_3_23/StatefulPartitionedCall:output:0-^dense_res_block_3_23/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_23/Identityy
Reshape_27/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_27/shape

Reshape_27Reshape&dense_res_block_3_23/Identity:output:0Reshape_27/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_27m
Identity_21IdentityMatMul_21:product:0*
T0*'
_output_shapes
:?????????2
Identity_21w
	MatMul_22MatMulIdentity_21:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_22
(inner_decoder_23/StatefulPartitionedCallStatefulPartitionedCallMatMul_22:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_23/StatefulPartitionedCallΣ
inner_decoder_23/IdentityIdentity1inner_decoder_23/StatefulPartitionedCall:output:0)^inner_decoder_23/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_23/Identity?
,dense_res_block_3_24/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_23/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_24/StatefulPartitionedCallγ
dense_res_block_3_24/IdentityIdentity5dense_res_block_3_24/StatefulPartitionedCall:output:0-^dense_res_block_3_24/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_24/Identityy
Reshape_28/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_28/shape

Reshape_28Reshape&dense_res_block_3_24/Identity:output:0Reshape_28/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_28m
Identity_22IdentityMatMul_22:product:0*
T0*'
_output_shapes
:?????????2
Identity_22w
	MatMul_23MatMulIdentity_22:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_23
(inner_decoder_24/StatefulPartitionedCallStatefulPartitionedCallMatMul_23:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_24/StatefulPartitionedCallΣ
inner_decoder_24/IdentityIdentity1inner_decoder_24/StatefulPartitionedCall:output:0)^inner_decoder_24/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_24/Identity?
,dense_res_block_3_25/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_24/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_25/StatefulPartitionedCallγ
dense_res_block_3_25/IdentityIdentity5dense_res_block_3_25/StatefulPartitionedCall:output:0-^dense_res_block_3_25/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_25/Identityy
Reshape_29/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_29/shape

Reshape_29Reshape&dense_res_block_3_25/Identity:output:0Reshape_29/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_29m
Identity_23IdentityMatMul_23:product:0*
T0*'
_output_shapes
:?????????2
Identity_23w
	MatMul_24MatMulIdentity_23:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_24
(inner_decoder_25/StatefulPartitionedCallStatefulPartitionedCallMatMul_24:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_25/StatefulPartitionedCallΣ
inner_decoder_25/IdentityIdentity1inner_decoder_25/StatefulPartitionedCall:output:0)^inner_decoder_25/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_25/Identity?
,dense_res_block_3_26/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_25/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_26/StatefulPartitionedCallγ
dense_res_block_3_26/IdentityIdentity5dense_res_block_3_26/StatefulPartitionedCall:output:0-^dense_res_block_3_26/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_26/Identityy
Reshape_30/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_30/shape

Reshape_30Reshape&dense_res_block_3_26/Identity:output:0Reshape_30/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_30m
Identity_24IdentityMatMul_24:product:0*
T0*'
_output_shapes
:?????????2
Identity_24w
	MatMul_25MatMulIdentity_24:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_25
(inner_decoder_26/StatefulPartitionedCallStatefulPartitionedCallMatMul_25:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_26/StatefulPartitionedCallΣ
inner_decoder_26/IdentityIdentity1inner_decoder_26/StatefulPartitionedCall:output:0)^inner_decoder_26/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_26/Identity?
,dense_res_block_3_27/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_26/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_27/StatefulPartitionedCallγ
dense_res_block_3_27/IdentityIdentity5dense_res_block_3_27/StatefulPartitionedCall:output:0-^dense_res_block_3_27/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_27/Identityy
Reshape_31/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_31/shape

Reshape_31Reshape&dense_res_block_3_27/Identity:output:0Reshape_31/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_31m
Identity_25IdentityMatMul_25:product:0*
T0*'
_output_shapes
:?????????2
Identity_25w
	MatMul_26MatMulIdentity_25:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_26
(inner_decoder_27/StatefulPartitionedCallStatefulPartitionedCallMatMul_26:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_27/StatefulPartitionedCallΣ
inner_decoder_27/IdentityIdentity1inner_decoder_27/StatefulPartitionedCall:output:0)^inner_decoder_27/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_27/Identity?
,dense_res_block_3_28/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_27/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_28/StatefulPartitionedCallγ
dense_res_block_3_28/IdentityIdentity5dense_res_block_3_28/StatefulPartitionedCall:output:0-^dense_res_block_3_28/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_28/Identityy
Reshape_32/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_32/shape

Reshape_32Reshape&dense_res_block_3_28/Identity:output:0Reshape_32/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_32m
Identity_26IdentityMatMul_26:product:0*
T0*'
_output_shapes
:?????????2
Identity_26w
	MatMul_27MatMulIdentity_26:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_27
(inner_decoder_28/StatefulPartitionedCallStatefulPartitionedCallMatMul_27:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_28/StatefulPartitionedCallΣ
inner_decoder_28/IdentityIdentity1inner_decoder_28/StatefulPartitionedCall:output:0)^inner_decoder_28/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_28/Identity?
,dense_res_block_3_29/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_28/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_29/StatefulPartitionedCallγ
dense_res_block_3_29/IdentityIdentity5dense_res_block_3_29/StatefulPartitionedCall:output:0-^dense_res_block_3_29/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_29/Identityy
Reshape_33/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_33/shape

Reshape_33Reshape&dense_res_block_3_29/Identity:output:0Reshape_33/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_33m
Identity_27IdentityMatMul_27:product:0*
T0*'
_output_shapes
:?????????2
Identity_27w
	MatMul_28MatMulIdentity_27:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_28
(inner_decoder_29/StatefulPartitionedCallStatefulPartitionedCallMatMul_28:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_29/StatefulPartitionedCallΣ
inner_decoder_29/IdentityIdentity1inner_decoder_29/StatefulPartitionedCall:output:0)^inner_decoder_29/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_29/Identity?
,dense_res_block_3_30/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_29/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_30/StatefulPartitionedCallγ
dense_res_block_3_30/IdentityIdentity5dense_res_block_3_30/StatefulPartitionedCall:output:0-^dense_res_block_3_30/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_30/Identityy
Reshape_34/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_34/shape

Reshape_34Reshape&dense_res_block_3_30/Identity:output:0Reshape_34/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_34m
Identity_28IdentityMatMul_28:product:0*
T0*'
_output_shapes
:?????????2
Identity_28w
	MatMul_29MatMulIdentity_28:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_29
(inner_decoder_30/StatefulPartitionedCallStatefulPartitionedCallMatMul_29:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_30/StatefulPartitionedCallΣ
inner_decoder_30/IdentityIdentity1inner_decoder_30/StatefulPartitionedCall:output:0)^inner_decoder_30/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_30/Identity?
,dense_res_block_3_31/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_30/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_31/StatefulPartitionedCallγ
dense_res_block_3_31/IdentityIdentity5dense_res_block_3_31/StatefulPartitionedCall:output:0-^dense_res_block_3_31/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_31/Identityy
Reshape_35/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_35/shape

Reshape_35Reshape&dense_res_block_3_31/Identity:output:0Reshape_35/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_35m
Identity_29IdentityMatMul_29:product:0*
T0*'
_output_shapes
:?????????2
Identity_29w
	MatMul_30MatMulIdentity_29:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_30
(inner_decoder_31/StatefulPartitionedCallStatefulPartitionedCallMatMul_30:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_31/StatefulPartitionedCallΣ
inner_decoder_31/IdentityIdentity1inner_decoder_31/StatefulPartitionedCall:output:0)^inner_decoder_31/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_31/Identity?
,dense_res_block_3_32/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_31/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_32/StatefulPartitionedCallγ
dense_res_block_3_32/IdentityIdentity5dense_res_block_3_32/StatefulPartitionedCall:output:0-^dense_res_block_3_32/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_32/Identityy
Reshape_36/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_36/shape

Reshape_36Reshape&dense_res_block_3_32/Identity:output:0Reshape_36/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_36m
Identity_30IdentityMatMul_30:product:0*
T0*'
_output_shapes
:?????????2
Identity_30w
	MatMul_31MatMulIdentity_30:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_31
(inner_decoder_32/StatefulPartitionedCallStatefulPartitionedCallMatMul_31:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_32/StatefulPartitionedCallΣ
inner_decoder_32/IdentityIdentity1inner_decoder_32/StatefulPartitionedCall:output:0)^inner_decoder_32/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_32/Identity?
,dense_res_block_3_33/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_32/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_33/StatefulPartitionedCallγ
dense_res_block_3_33/IdentityIdentity5dense_res_block_3_33/StatefulPartitionedCall:output:0-^dense_res_block_3_33/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_33/Identityy
Reshape_37/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_37/shape

Reshape_37Reshape&dense_res_block_3_33/Identity:output:0Reshape_37/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_37m
Identity_31IdentityMatMul_31:product:0*
T0*'
_output_shapes
:?????????2
Identity_31w
	MatMul_32MatMulIdentity_31:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_32
(inner_decoder_33/StatefulPartitionedCallStatefulPartitionedCallMatMul_32:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_33/StatefulPartitionedCallΣ
inner_decoder_33/IdentityIdentity1inner_decoder_33/StatefulPartitionedCall:output:0)^inner_decoder_33/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_33/Identity?
,dense_res_block_3_34/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_33/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_34/StatefulPartitionedCallγ
dense_res_block_3_34/IdentityIdentity5dense_res_block_3_34/StatefulPartitionedCall:output:0-^dense_res_block_3_34/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_34/Identityy
Reshape_38/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_38/shape

Reshape_38Reshape&dense_res_block_3_34/Identity:output:0Reshape_38/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_38m
Identity_32IdentityMatMul_32:product:0*
T0*'
_output_shapes
:?????????2
Identity_32w
	MatMul_33MatMulIdentity_32:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_33
(inner_decoder_34/StatefulPartitionedCallStatefulPartitionedCallMatMul_33:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_34/StatefulPartitionedCallΣ
inner_decoder_34/IdentityIdentity1inner_decoder_34/StatefulPartitionedCall:output:0)^inner_decoder_34/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_34/Identity?
,dense_res_block_3_35/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_34/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_35/StatefulPartitionedCallγ
dense_res_block_3_35/IdentityIdentity5dense_res_block_3_35/StatefulPartitionedCall:output:0-^dense_res_block_3_35/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_35/Identityy
Reshape_39/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_39/shape

Reshape_39Reshape&dense_res_block_3_35/Identity:output:0Reshape_39/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_39m
Identity_33IdentityMatMul_33:product:0*
T0*'
_output_shapes
:?????????2
Identity_33w
	MatMul_34MatMulIdentity_33:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_34
(inner_decoder_35/StatefulPartitionedCallStatefulPartitionedCallMatMul_34:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_35/StatefulPartitionedCallΣ
inner_decoder_35/IdentityIdentity1inner_decoder_35/StatefulPartitionedCall:output:0)^inner_decoder_35/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_35/Identity?
,dense_res_block_3_36/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_35/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_36/StatefulPartitionedCallγ
dense_res_block_3_36/IdentityIdentity5dense_res_block_3_36/StatefulPartitionedCall:output:0-^dense_res_block_3_36/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_36/Identityy
Reshape_40/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_40/shape

Reshape_40Reshape&dense_res_block_3_36/Identity:output:0Reshape_40/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_40m
Identity_34IdentityMatMul_34:product:0*
T0*'
_output_shapes
:?????????2
Identity_34w
	MatMul_35MatMulIdentity_34:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_35
(inner_decoder_36/StatefulPartitionedCallStatefulPartitionedCallMatMul_35:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_36/StatefulPartitionedCallΣ
inner_decoder_36/IdentityIdentity1inner_decoder_36/StatefulPartitionedCall:output:0)^inner_decoder_36/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_36/Identity?
,dense_res_block_3_37/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_36/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_37/StatefulPartitionedCallγ
dense_res_block_3_37/IdentityIdentity5dense_res_block_3_37/StatefulPartitionedCall:output:0-^dense_res_block_3_37/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_37/Identityy
Reshape_41/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_41/shape

Reshape_41Reshape&dense_res_block_3_37/Identity:output:0Reshape_41/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_41m
Identity_35IdentityMatMul_35:product:0*
T0*'
_output_shapes
:?????????2
Identity_35w
	MatMul_36MatMulIdentity_35:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_36
(inner_decoder_37/StatefulPartitionedCallStatefulPartitionedCallMatMul_36:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_37/StatefulPartitionedCallΣ
inner_decoder_37/IdentityIdentity1inner_decoder_37/StatefulPartitionedCall:output:0)^inner_decoder_37/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_37/Identity?
,dense_res_block_3_38/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_37/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_38/StatefulPartitionedCallγ
dense_res_block_3_38/IdentityIdentity5dense_res_block_3_38/StatefulPartitionedCall:output:0-^dense_res_block_3_38/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_38/Identityy
Reshape_42/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_42/shape

Reshape_42Reshape&dense_res_block_3_38/Identity:output:0Reshape_42/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_42m
Identity_36IdentityMatMul_36:product:0*
T0*'
_output_shapes
:?????????2
Identity_36w
	MatMul_37MatMulIdentity_36:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_37
(inner_decoder_38/StatefulPartitionedCallStatefulPartitionedCallMatMul_37:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_38/StatefulPartitionedCallΣ
inner_decoder_38/IdentityIdentity1inner_decoder_38/StatefulPartitionedCall:output:0)^inner_decoder_38/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_38/Identity?
,dense_res_block_3_39/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_38/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_39/StatefulPartitionedCallγ
dense_res_block_3_39/IdentityIdentity5dense_res_block_3_39/StatefulPartitionedCall:output:0-^dense_res_block_3_39/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_39/Identityy
Reshape_43/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_43/shape

Reshape_43Reshape&dense_res_block_3_39/Identity:output:0Reshape_43/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_43m
Identity_37IdentityMatMul_37:product:0*
T0*'
_output_shapes
:?????????2
Identity_37w
	MatMul_38MatMulIdentity_37:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_38
(inner_decoder_39/StatefulPartitionedCallStatefulPartitionedCallMatMul_38:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_39/StatefulPartitionedCallΣ
inner_decoder_39/IdentityIdentity1inner_decoder_39/StatefulPartitionedCall:output:0)^inner_decoder_39/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_39/Identity?
,dense_res_block_3_40/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_39/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_40/StatefulPartitionedCallγ
dense_res_block_3_40/IdentityIdentity5dense_res_block_3_40/StatefulPartitionedCall:output:0-^dense_res_block_3_40/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_40/Identityy
Reshape_44/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_44/shape

Reshape_44Reshape&dense_res_block_3_40/Identity:output:0Reshape_44/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_44m
Identity_38IdentityMatMul_38:product:0*
T0*'
_output_shapes
:?????????2
Identity_38w
	MatMul_39MatMulIdentity_38:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_39
(inner_decoder_40/StatefulPartitionedCallStatefulPartitionedCallMatMul_39:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_40/StatefulPartitionedCallΣ
inner_decoder_40/IdentityIdentity1inner_decoder_40/StatefulPartitionedCall:output:0)^inner_decoder_40/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_40/Identity?
,dense_res_block_3_41/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_40/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_41/StatefulPartitionedCallγ
dense_res_block_3_41/IdentityIdentity5dense_res_block_3_41/StatefulPartitionedCall:output:0-^dense_res_block_3_41/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_41/Identityy
Reshape_45/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_45/shape

Reshape_45Reshape&dense_res_block_3_41/Identity:output:0Reshape_45/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_45m
Identity_39IdentityMatMul_39:product:0*
T0*'
_output_shapes
:?????????2
Identity_39w
	MatMul_40MatMulIdentity_39:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_40
(inner_decoder_41/StatefulPartitionedCallStatefulPartitionedCallMatMul_40:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_41/StatefulPartitionedCallΣ
inner_decoder_41/IdentityIdentity1inner_decoder_41/StatefulPartitionedCall:output:0)^inner_decoder_41/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_41/Identity?
,dense_res_block_3_42/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_41/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_42/StatefulPartitionedCallγ
dense_res_block_3_42/IdentityIdentity5dense_res_block_3_42/StatefulPartitionedCall:output:0-^dense_res_block_3_42/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_42/Identityy
Reshape_46/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_46/shape

Reshape_46Reshape&dense_res_block_3_42/Identity:output:0Reshape_46/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_46m
Identity_40IdentityMatMul_40:product:0*
T0*'
_output_shapes
:?????????2
Identity_40w
	MatMul_41MatMulIdentity_40:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_41
(inner_decoder_42/StatefulPartitionedCallStatefulPartitionedCallMatMul_41:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_42/StatefulPartitionedCallΣ
inner_decoder_42/IdentityIdentity1inner_decoder_42/StatefulPartitionedCall:output:0)^inner_decoder_42/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_42/Identity?
,dense_res_block_3_43/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_42/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_43/StatefulPartitionedCallγ
dense_res_block_3_43/IdentityIdentity5dense_res_block_3_43/StatefulPartitionedCall:output:0-^dense_res_block_3_43/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_43/Identityy
Reshape_47/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_47/shape

Reshape_47Reshape&dense_res_block_3_43/Identity:output:0Reshape_47/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_47m
Identity_41IdentityMatMul_41:product:0*
T0*'
_output_shapes
:?????????2
Identity_41w
	MatMul_42MatMulIdentity_41:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_42
(inner_decoder_43/StatefulPartitionedCallStatefulPartitionedCallMatMul_42:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_43/StatefulPartitionedCallΣ
inner_decoder_43/IdentityIdentity1inner_decoder_43/StatefulPartitionedCall:output:0)^inner_decoder_43/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_43/Identity?
,dense_res_block_3_44/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_43/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_44/StatefulPartitionedCallγ
dense_res_block_3_44/IdentityIdentity5dense_res_block_3_44/StatefulPartitionedCall:output:0-^dense_res_block_3_44/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_44/Identityy
Reshape_48/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_48/shape

Reshape_48Reshape&dense_res_block_3_44/Identity:output:0Reshape_48/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_48m
Identity_42IdentityMatMul_42:product:0*
T0*'
_output_shapes
:?????????2
Identity_42w
	MatMul_43MatMulIdentity_42:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_43
(inner_decoder_44/StatefulPartitionedCallStatefulPartitionedCallMatMul_43:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_44/StatefulPartitionedCallΣ
inner_decoder_44/IdentityIdentity1inner_decoder_44/StatefulPartitionedCall:output:0)^inner_decoder_44/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_44/Identity?
,dense_res_block_3_45/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_44/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_45/StatefulPartitionedCallγ
dense_res_block_3_45/IdentityIdentity5dense_res_block_3_45/StatefulPartitionedCall:output:0-^dense_res_block_3_45/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_45/Identityy
Reshape_49/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_49/shape

Reshape_49Reshape&dense_res_block_3_45/Identity:output:0Reshape_49/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_49m
Identity_43IdentityMatMul_43:product:0*
T0*'
_output_shapes
:?????????2
Identity_43w
	MatMul_44MatMulIdentity_43:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_44
(inner_decoder_45/StatefulPartitionedCallStatefulPartitionedCallMatMul_44:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_45/StatefulPartitionedCallΣ
inner_decoder_45/IdentityIdentity1inner_decoder_45/StatefulPartitionedCall:output:0)^inner_decoder_45/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_45/Identity?
,dense_res_block_3_46/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_45/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_46/StatefulPartitionedCallγ
dense_res_block_3_46/IdentityIdentity5dense_res_block_3_46/StatefulPartitionedCall:output:0-^dense_res_block_3_46/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_46/Identityy
Reshape_50/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_50/shape

Reshape_50Reshape&dense_res_block_3_46/Identity:output:0Reshape_50/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_50m
Identity_44IdentityMatMul_44:product:0*
T0*'
_output_shapes
:?????????2
Identity_44w
	MatMul_45MatMulIdentity_44:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_45
(inner_decoder_46/StatefulPartitionedCallStatefulPartitionedCallMatMul_45:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_46/StatefulPartitionedCallΣ
inner_decoder_46/IdentityIdentity1inner_decoder_46/StatefulPartitionedCall:output:0)^inner_decoder_46/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_46/Identity?
,dense_res_block_3_47/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_46/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_47/StatefulPartitionedCallγ
dense_res_block_3_47/IdentityIdentity5dense_res_block_3_47/StatefulPartitionedCall:output:0-^dense_res_block_3_47/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_47/Identityy
Reshape_51/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_51/shape

Reshape_51Reshape&dense_res_block_3_47/Identity:output:0Reshape_51/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_51m
Identity_45IdentityMatMul_45:product:0*
T0*'
_output_shapes
:?????????2
Identity_45w
	MatMul_46MatMulIdentity_45:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_46
(inner_decoder_47/StatefulPartitionedCallStatefulPartitionedCallMatMul_46:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_47/StatefulPartitionedCallΣ
inner_decoder_47/IdentityIdentity1inner_decoder_47/StatefulPartitionedCall:output:0)^inner_decoder_47/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_47/Identity?
,dense_res_block_3_48/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_47/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_48/StatefulPartitionedCallγ
dense_res_block_3_48/IdentityIdentity5dense_res_block_3_48/StatefulPartitionedCall:output:0-^dense_res_block_3_48/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_48/Identityy
Reshape_52/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_52/shape

Reshape_52Reshape&dense_res_block_3_48/Identity:output:0Reshape_52/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_52m
Identity_46IdentityMatMul_46:product:0*
T0*'
_output_shapes
:?????????2
Identity_46w
	MatMul_47MatMulIdentity_46:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_47
(inner_decoder_48/StatefulPartitionedCallStatefulPartitionedCallMatMul_47:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_48/StatefulPartitionedCallΣ
inner_decoder_48/IdentityIdentity1inner_decoder_48/StatefulPartitionedCall:output:0)^inner_decoder_48/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_48/Identity?
,dense_res_block_3_49/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_48/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_49/StatefulPartitionedCallγ
dense_res_block_3_49/IdentityIdentity5dense_res_block_3_49/StatefulPartitionedCall:output:0-^dense_res_block_3_49/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_49/Identityy
Reshape_53/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_53/shape

Reshape_53Reshape&dense_res_block_3_49/Identity:output:0Reshape_53/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_53m
Identity_47IdentityMatMul_47:product:0*
T0*'
_output_shapes
:?????????2
Identity_47w
	MatMul_48MatMulIdentity_47:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_48
(inner_decoder_49/StatefulPartitionedCallStatefulPartitionedCallMatMul_48:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_49/StatefulPartitionedCallΣ
inner_decoder_49/IdentityIdentity1inner_decoder_49/StatefulPartitionedCall:output:0)^inner_decoder_49/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_49/Identity?
,dense_res_block_3_50/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_49/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_50/StatefulPartitionedCallγ
dense_res_block_3_50/IdentityIdentity5dense_res_block_3_50/StatefulPartitionedCall:output:0-^dense_res_block_3_50/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_50/Identityy
Reshape_54/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_54/shape

Reshape_54Reshape&dense_res_block_3_50/Identity:output:0Reshape_54/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_54m
Identity_48IdentityMatMul_48:product:0*
T0*'
_output_shapes
:?????????2
Identity_48w
	MatMul_49MatMulIdentity_48:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_49
(inner_decoder_50/StatefulPartitionedCallStatefulPartitionedCallMatMul_49:product:0inner_decoder_5929787*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132*
(inner_decoder_50/StatefulPartitionedCallΣ
inner_decoder_50/IdentityIdentity1inner_decoder_50/StatefulPartitionedCall:output:0)^inner_decoder_50/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
inner_decoder_50/Identity?
,dense_res_block_3_51/StatefulPartitionedCallStatefulPartitionedCall"inner_decoder_50/Identity:output:0dense_res_block_3_5929902dense_res_block_3_5929904dense_res_block_3_5929906dense_res_block_3_5929908dense_res_block_3_5929910dense_res_block_3_5929912dense_res_block_3_5929914dense_res_block_3_5929916dense_res_block_3_5929918dense_res_block_3_5929920*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332.
,dense_res_block_3_51/StatefulPartitionedCallγ
dense_res_block_3_51/IdentityIdentity5dense_res_block_3_51/StatefulPartitionedCall:output:0-^dense_res_block_3_51/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_3_51/Identityy
Reshape_55/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_55/shape

Reshape_55Reshape&dense_res_block_3_51/Identity:output:0Reshape_55/shape:output:0*
T0*,
_output_shapes
:?????????2

Reshape_55m
Identity_49IdentityMatMul_49:product:0*
T0*'
_output_shapes
:?????????2
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
:?????????22

concat_2
+dense_res_block_2_2/StatefulPartitionedCallStatefulPartitionedCallReshape_2:output:0dense_res_block_2_5929713dense_res_block_2_5929715dense_res_block_2_5929717dense_res_block_2_5929719dense_res_block_2_5929721dense_res_block_2_5929723dense_res_block_2_5929725dense_res_block_2_5929727dense_res_block_2_5929729dense_res_block_2_5929731*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_2_layer_call_and_return_conditional_losses_168567832-
+dense_res_block_2_2/StatefulPartitionedCallί
dense_res_block_2_2/IdentityIdentity4dense_res_block_2_2/StatefulPartitionedCall:output:0,^dense_res_block_2_2/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_2_2/Identity
'inner_encoder_2/StatefulPartitionedCallStatefulPartitionedCall%dense_res_block_2_2/Identity:output:0inner_encoder_5929759*
Tin
2*
Tout
2*'
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_encoder_layer_call_and_return_conditional_losses_168574512)
'inner_encoder_2/StatefulPartitionedCallΞ
inner_encoder_2/IdentityIdentity0inner_encoder_2/StatefulPartitionedCall:output:0(^inner_encoder_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
inner_encoder_2/Identity
	MatMul_50MatMul!inner_encoder_2/Identity:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_50m
Identity_50IdentityMatMul_50:product:0*
T0*'
_output_shapes
:?????????2
Identity_50y
Reshape_56/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_56/shape

Reshape_56ReshapeIdentity_50:output:0Reshape_56/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_56w
	MatMul_51MatMulIdentity_50:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_51m
Identity_51IdentityMatMul_51:product:0*
T0*'
_output_shapes
:?????????2
Identity_51y
Reshape_57/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_57/shape

Reshape_57ReshapeIdentity_51:output:0Reshape_57/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_57w
	MatMul_52MatMulIdentity_51:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_52m
Identity_52IdentityMatMul_52:product:0*
T0*'
_output_shapes
:?????????2
Identity_52y
Reshape_58/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_58/shape

Reshape_58ReshapeIdentity_52:output:0Reshape_58/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_58w
	MatMul_53MatMulIdentity_52:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_53m
Identity_53IdentityMatMul_53:product:0*
T0*'
_output_shapes
:?????????2
Identity_53y
Reshape_59/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_59/shape

Reshape_59ReshapeIdentity_53:output:0Reshape_59/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_59w
	MatMul_54MatMulIdentity_53:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_54m
Identity_54IdentityMatMul_54:product:0*
T0*'
_output_shapes
:?????????2
Identity_54y
Reshape_60/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_60/shape

Reshape_60ReshapeIdentity_54:output:0Reshape_60/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_60w
	MatMul_55MatMulIdentity_54:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_55m
Identity_55IdentityMatMul_55:product:0*
T0*'
_output_shapes
:?????????2
Identity_55y
Reshape_61/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_61/shape

Reshape_61ReshapeIdentity_55:output:0Reshape_61/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_61w
	MatMul_56MatMulIdentity_55:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_56m
Identity_56IdentityMatMul_56:product:0*
T0*'
_output_shapes
:?????????2
Identity_56y
Reshape_62/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_62/shape

Reshape_62ReshapeIdentity_56:output:0Reshape_62/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_62w
	MatMul_57MatMulIdentity_56:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_57m
Identity_57IdentityMatMul_57:product:0*
T0*'
_output_shapes
:?????????2
Identity_57y
Reshape_63/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_63/shape

Reshape_63ReshapeIdentity_57:output:0Reshape_63/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_63w
	MatMul_58MatMulIdentity_57:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_58m
Identity_58IdentityMatMul_58:product:0*
T0*'
_output_shapes
:?????????2
Identity_58y
Reshape_64/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_64/shape

Reshape_64ReshapeIdentity_58:output:0Reshape_64/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_64w
	MatMul_59MatMulIdentity_58:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_59m
Identity_59IdentityMatMul_59:product:0*
T0*'
_output_shapes
:?????????2
Identity_59y
Reshape_65/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_65/shape

Reshape_65ReshapeIdentity_59:output:0Reshape_65/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_65w
	MatMul_60MatMulIdentity_59:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_60m
Identity_60IdentityMatMul_60:product:0*
T0*'
_output_shapes
:?????????2
Identity_60y
Reshape_66/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_66/shape

Reshape_66ReshapeIdentity_60:output:0Reshape_66/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_66w
	MatMul_61MatMulIdentity_60:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_61m
Identity_61IdentityMatMul_61:product:0*
T0*'
_output_shapes
:?????????2
Identity_61y
Reshape_67/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_67/shape

Reshape_67ReshapeIdentity_61:output:0Reshape_67/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_67w
	MatMul_62MatMulIdentity_61:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_62m
Identity_62IdentityMatMul_62:product:0*
T0*'
_output_shapes
:?????????2
Identity_62y
Reshape_68/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_68/shape

Reshape_68ReshapeIdentity_62:output:0Reshape_68/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_68w
	MatMul_63MatMulIdentity_62:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_63m
Identity_63IdentityMatMul_63:product:0*
T0*'
_output_shapes
:?????????2
Identity_63y
Reshape_69/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_69/shape

Reshape_69ReshapeIdentity_63:output:0Reshape_69/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_69w
	MatMul_64MatMulIdentity_63:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_64m
Identity_64IdentityMatMul_64:product:0*
T0*'
_output_shapes
:?????????2
Identity_64y
Reshape_70/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_70/shape

Reshape_70ReshapeIdentity_64:output:0Reshape_70/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_70w
	MatMul_65MatMulIdentity_64:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_65m
Identity_65IdentityMatMul_65:product:0*
T0*'
_output_shapes
:?????????2
Identity_65y
Reshape_71/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_71/shape

Reshape_71ReshapeIdentity_65:output:0Reshape_71/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_71w
	MatMul_66MatMulIdentity_65:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_66m
Identity_66IdentityMatMul_66:product:0*
T0*'
_output_shapes
:?????????2
Identity_66y
Reshape_72/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_72/shape

Reshape_72ReshapeIdentity_66:output:0Reshape_72/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_72w
	MatMul_67MatMulIdentity_66:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_67m
Identity_67IdentityMatMul_67:product:0*
T0*'
_output_shapes
:?????????2
Identity_67y
Reshape_73/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_73/shape

Reshape_73ReshapeIdentity_67:output:0Reshape_73/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_73w
	MatMul_68MatMulIdentity_67:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_68m
Identity_68IdentityMatMul_68:product:0*
T0*'
_output_shapes
:?????????2
Identity_68y
Reshape_74/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_74/shape

Reshape_74ReshapeIdentity_68:output:0Reshape_74/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_74w
	MatMul_69MatMulIdentity_68:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_69m
Identity_69IdentityMatMul_69:product:0*
T0*'
_output_shapes
:?????????2
Identity_69y
Reshape_75/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_75/shape

Reshape_75ReshapeIdentity_69:output:0Reshape_75/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_75w
	MatMul_70MatMulIdentity_69:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_70m
Identity_70IdentityMatMul_70:product:0*
T0*'
_output_shapes
:?????????2
Identity_70y
Reshape_76/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_76/shape

Reshape_76ReshapeIdentity_70:output:0Reshape_76/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_76w
	MatMul_71MatMulIdentity_70:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_71m
Identity_71IdentityMatMul_71:product:0*
T0*'
_output_shapes
:?????????2
Identity_71y
Reshape_77/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_77/shape

Reshape_77ReshapeIdentity_71:output:0Reshape_77/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_77w
	MatMul_72MatMulIdentity_71:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_72m
Identity_72IdentityMatMul_72:product:0*
T0*'
_output_shapes
:?????????2
Identity_72y
Reshape_78/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_78/shape

Reshape_78ReshapeIdentity_72:output:0Reshape_78/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_78w
	MatMul_73MatMulIdentity_72:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_73m
Identity_73IdentityMatMul_73:product:0*
T0*'
_output_shapes
:?????????2
Identity_73y
Reshape_79/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_79/shape

Reshape_79ReshapeIdentity_73:output:0Reshape_79/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_79w
	MatMul_74MatMulIdentity_73:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_74m
Identity_74IdentityMatMul_74:product:0*
T0*'
_output_shapes
:?????????2
Identity_74y
Reshape_80/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_80/shape

Reshape_80ReshapeIdentity_74:output:0Reshape_80/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_80w
	MatMul_75MatMulIdentity_74:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_75m
Identity_75IdentityMatMul_75:product:0*
T0*'
_output_shapes
:?????????2
Identity_75y
Reshape_81/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_81/shape

Reshape_81ReshapeIdentity_75:output:0Reshape_81/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_81w
	MatMul_76MatMulIdentity_75:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_76m
Identity_76IdentityMatMul_76:product:0*
T0*'
_output_shapes
:?????????2
Identity_76y
Reshape_82/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_82/shape

Reshape_82ReshapeIdentity_76:output:0Reshape_82/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_82w
	MatMul_77MatMulIdentity_76:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_77m
Identity_77IdentityMatMul_77:product:0*
T0*'
_output_shapes
:?????????2
Identity_77y
Reshape_83/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_83/shape

Reshape_83ReshapeIdentity_77:output:0Reshape_83/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_83w
	MatMul_78MatMulIdentity_77:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_78m
Identity_78IdentityMatMul_78:product:0*
T0*'
_output_shapes
:?????????2
Identity_78y
Reshape_84/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_84/shape

Reshape_84ReshapeIdentity_78:output:0Reshape_84/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_84w
	MatMul_79MatMulIdentity_78:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_79m
Identity_79IdentityMatMul_79:product:0*
T0*'
_output_shapes
:?????????2
Identity_79y
Reshape_85/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_85/shape

Reshape_85ReshapeIdentity_79:output:0Reshape_85/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_85w
	MatMul_80MatMulIdentity_79:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_80m
Identity_80IdentityMatMul_80:product:0*
T0*'
_output_shapes
:?????????2
Identity_80y
Reshape_86/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_86/shape

Reshape_86ReshapeIdentity_80:output:0Reshape_86/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_86w
	MatMul_81MatMulIdentity_80:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_81m
Identity_81IdentityMatMul_81:product:0*
T0*'
_output_shapes
:?????????2
Identity_81y
Reshape_87/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_87/shape

Reshape_87ReshapeIdentity_81:output:0Reshape_87/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_87w
	MatMul_82MatMulIdentity_81:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_82m
Identity_82IdentityMatMul_82:product:0*
T0*'
_output_shapes
:?????????2
Identity_82y
Reshape_88/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_88/shape

Reshape_88ReshapeIdentity_82:output:0Reshape_88/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_88w
	MatMul_83MatMulIdentity_82:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_83m
Identity_83IdentityMatMul_83:product:0*
T0*'
_output_shapes
:?????????2
Identity_83y
Reshape_89/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_89/shape

Reshape_89ReshapeIdentity_83:output:0Reshape_89/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_89w
	MatMul_84MatMulIdentity_83:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_84m
Identity_84IdentityMatMul_84:product:0*
T0*'
_output_shapes
:?????????2
Identity_84y
Reshape_90/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_90/shape

Reshape_90ReshapeIdentity_84:output:0Reshape_90/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_90w
	MatMul_85MatMulIdentity_84:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_85m
Identity_85IdentityMatMul_85:product:0*
T0*'
_output_shapes
:?????????2
Identity_85y
Reshape_91/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_91/shape

Reshape_91ReshapeIdentity_85:output:0Reshape_91/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_91w
	MatMul_86MatMulIdentity_85:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_86m
Identity_86IdentityMatMul_86:product:0*
T0*'
_output_shapes
:?????????2
Identity_86y
Reshape_92/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_92/shape

Reshape_92ReshapeIdentity_86:output:0Reshape_92/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_92w
	MatMul_87MatMulIdentity_86:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_87m
Identity_87IdentityMatMul_87:product:0*
T0*'
_output_shapes
:?????????2
Identity_87y
Reshape_93/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_93/shape

Reshape_93ReshapeIdentity_87:output:0Reshape_93/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_93w
	MatMul_88MatMulIdentity_87:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_88m
Identity_88IdentityMatMul_88:product:0*
T0*'
_output_shapes
:?????????2
Identity_88y
Reshape_94/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_94/shape

Reshape_94ReshapeIdentity_88:output:0Reshape_94/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_94w
	MatMul_89MatMulIdentity_88:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_89m
Identity_89IdentityMatMul_89:product:0*
T0*'
_output_shapes
:?????????2
Identity_89y
Reshape_95/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_95/shape

Reshape_95ReshapeIdentity_89:output:0Reshape_95/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_95w
	MatMul_90MatMulIdentity_89:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_90m
Identity_90IdentityMatMul_90:product:0*
T0*'
_output_shapes
:?????????2
Identity_90y
Reshape_96/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_96/shape

Reshape_96ReshapeIdentity_90:output:0Reshape_96/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_96w
	MatMul_91MatMulIdentity_90:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_91m
Identity_91IdentityMatMul_91:product:0*
T0*'
_output_shapes
:?????????2
Identity_91y
Reshape_97/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_97/shape

Reshape_97ReshapeIdentity_91:output:0Reshape_97/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_97w
	MatMul_92MatMulIdentity_91:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_92m
Identity_92IdentityMatMul_92:product:0*
T0*'
_output_shapes
:?????????2
Identity_92y
Reshape_98/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_98/shape

Reshape_98ReshapeIdentity_92:output:0Reshape_98/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_98w
	MatMul_93MatMulIdentity_92:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_93m
Identity_93IdentityMatMul_93:product:0*
T0*'
_output_shapes
:?????????2
Identity_93y
Reshape_99/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_99/shape

Reshape_99ReshapeIdentity_93:output:0Reshape_99/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_99w
	MatMul_94MatMulIdentity_93:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_94m
Identity_94IdentityMatMul_94:product:0*
T0*'
_output_shapes
:?????????2
Identity_94{
Reshape_100/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_100/shape
Reshape_100ReshapeIdentity_94:output:0Reshape_100/shape:output:0*
T0*+
_output_shapes
:?????????2
Reshape_100w
	MatMul_95MatMulIdentity_94:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_95m
Identity_95IdentityMatMul_95:product:0*
T0*'
_output_shapes
:?????????2
Identity_95{
Reshape_101/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_101/shape
Reshape_101ReshapeIdentity_95:output:0Reshape_101/shape:output:0*
T0*+
_output_shapes
:?????????2
Reshape_101w
	MatMul_96MatMulIdentity_95:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_96m
Identity_96IdentityMatMul_96:product:0*
T0*'
_output_shapes
:?????????2
Identity_96{
Reshape_102/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_102/shape
Reshape_102ReshapeIdentity_96:output:0Reshape_102/shape:output:0*
T0*+
_output_shapes
:?????????2
Reshape_102w
	MatMul_97MatMulIdentity_96:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_97m
Identity_97IdentityMatMul_97:product:0*
T0*'
_output_shapes
:?????????2
Identity_97{
Reshape_103/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_103/shape
Reshape_103ReshapeIdentity_97:output:0Reshape_103/shape:output:0*
T0*+
_output_shapes
:?????????2
Reshape_103w
	MatMul_98MatMulIdentity_97:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_98m
Identity_98IdentityMatMul_98:product:0*
T0*'
_output_shapes
:?????????2
Identity_98{
Reshape_104/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_104/shape
Reshape_104ReshapeIdentity_98:output:0Reshape_104/shape:output:0*
T0*+
_output_shapes
:?????????2
Reshape_104w
	MatMul_99MatMulIdentity_98:output:0diag:output:0*
T0*'
_output_shapes
:?????????2
	MatMul_99m
Identity_99IdentityMatMul_99:product:0*
T0*'
_output_shapes
:?????????2
Identity_99{
Reshape_105/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_105/shape
Reshape_105ReshapeIdentity_99:output:0Reshape_105/shape:output:0*
T0*+
_output_shapes
:?????????2
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
:?????????22

concat_3
+dense_res_block_2_3/StatefulPartitionedCallStatefulPartitionedCallReshape_3:output:0dense_res_block_2_5929713dense_res_block_2_5929715dense_res_block_2_5929717dense_res_block_2_5929719dense_res_block_2_5929721dense_res_block_2_5929723dense_res_block_2_5929725dense_res_block_2_5929727dense_res_block_2_5929729dense_res_block_2_5929731*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_2_layer_call_and_return_conditional_losses_168567832-
+dense_res_block_2_3/StatefulPartitionedCallί
dense_res_block_2_3/IdentityIdentity4dense_res_block_2_3/StatefulPartitionedCall:output:0,^dense_res_block_2_3/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
dense_res_block_2_3/Identity
'inner_encoder_3/StatefulPartitionedCallStatefulPartitionedCall%dense_res_block_2_3/Identity:output:0inner_encoder_5929759*
Tin
2*
Tout
2*'
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_encoder_layer_call_and_return_conditional_losses_168574512)
'inner_encoder_3/StatefulPartitionedCallΞ
inner_encoder_3/IdentityIdentity0inner_encoder_3/StatefulPartitionedCall:output:0(^inner_encoder_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
inner_encoder_3/Identity{
Reshape_106/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????2      2
Reshape_106/shape
Reshape_106Reshape!inner_encoder_3/Identity:output:0Reshape_106/shape:output:0*
T0*+
_output_shapes
:?????????22
Reshape_106
RelMSE_1/subSubconcat_3:output:0Reshape_106:output:0*
T0*+
_output_shapes
:?????????22
RelMSE_1/subt
RelMSE_1/SquareSquareRelMSE_1/sub:z:0*
T0*+
_output_shapes
:?????????22
RelMSE_1/Square
RelMSE_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
RelMSE_1/Mean/reduction_indices
RelMSE_1/MeanMeanRelMSE_1/Square:y:0(RelMSE_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????22
RelMSE_1/Mean|
RelMSE_1/Square_1SquareReshape_106:output:0*
T0*+
_output_shapes
:?????????22
RelMSE_1/Square_1
!RelMSE_1/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!RelMSE_1/Mean_1/reduction_indices
RelMSE_1/Mean_1MeanRelMSE_1/Square_1:y:0*RelMSE_1/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????22
RelMSE_1/Mean_1e
RelMSE_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'72
RelMSE_1/add/y
RelMSE_1/addAddV2RelMSE_1/Mean_1:output:0RelMSE_1/add/y:output:0*
T0*'
_output_shapes
:?????????22
RelMSE_1/add
RelMSE_1/truedivRealDivRelMSE_1/Mean:output:0RelMSE_1/add:z:0*
T0*'
_output_shapes
:?????????22
RelMSE_1/truediv
!RelMSE_1/Mean_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!RelMSE_1/Mean_2/reduction_indices
RelMSE_1/Mean_2MeanRelMSE_1/truediv:z:0*RelMSE_1/Mean_2/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
RelMSE_1/Mean_2
RelMSE_1/weighted_loss/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
RelMSE_1/weighted_loss/Cast/x―
RelMSE_1/weighted_loss/MulMulRelMSE_1/Mean_2:output:0&RelMSE_1/weighted_loss/Cast/x:output:0*
T0*#
_output_shapes
:?????????2
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
#RelMSE_1/weighted_loss/num_elementsΊ
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
RelMSE_1/weighted_loss/Sum_1Ύ
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
mul_1ώ
Onetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_res_block_2_5929713* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_2/hidden0/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_2/hidden0/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/addώ
Onetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_res_block_2_5929717* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_2/hidden1/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_2/hidden1/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/addώ
Onetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_res_block_2_5929721* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_2/hidden2/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_2/hidden2/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/addώ
Onetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_res_block_2_5929725* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_2/hidden3/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_2/hidden3/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/addό
Nnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_res_block_2_5929729* 
_output_shapes
:
*
dtype02P
Nnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Square/ReadVariableOp
?network_arch/dense_res_block_2/output/kernel/Regularizer/SquareSquareVnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2A
?network_arch/dense_res_block_2/output/kernel/Regularizer/SquareΡ
>network_arch/dense_res_block_2/output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2@
>network_arch/dense_res_block_2/output/kernel/Regularizer/Const²
<network_arch/dense_res_block_2/output/kernel/Regularizer/SumSumCnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Square:y:0Gnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_2/output/kernel/Regularizer/SumΕ
>network_arch/dense_res_block_2/output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22@
>network_arch/dense_res_block_2/output/kernel/Regularizer/mul/x΄
<network_arch/dense_res_block_2/output/kernel/Regularizer/mulMulGnetwork_arch/dense_res_block_2/output/kernel/Regularizer/mul/x:output:0Enetwork_arch/dense_res_block_2/output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_2/output/kernel/Regularizer/mulΕ
>network_arch/dense_res_block_2/output/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>network_arch/dense_res_block_2/output/kernel/Regularizer/add/x±
<network_arch/dense_res_block_2/output/kernel/Regularizer/addAddV2Gnetwork_arch/dense_res_block_2/output/kernel/Regularizer/add/x:output:0@network_arch/dense_res_block_2/output/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_2/output/kernel/Regularizer/addώ
Onetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_res_block_3_5929902* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_3/hidden0/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_3/hidden0/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/addώ
Onetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_res_block_3_5929906* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_3/hidden1/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_3/hidden1/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/addώ
Onetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_res_block_3_5929910* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_3/hidden2/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_3/hidden2/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/addώ
Onetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_res_block_3_5929914* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_3/hidden3/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_3/hidden3/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/addό
Nnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_res_block_3_5929918* 
_output_shapes
:
*
dtype02P
Nnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Square/ReadVariableOp
?network_arch/dense_res_block_3/output/kernel/Regularizer/SquareSquareVnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2A
?network_arch/dense_res_block_3/output/kernel/Regularizer/SquareΡ
>network_arch/dense_res_block_3/output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2@
>network_arch/dense_res_block_3/output/kernel/Regularizer/Const²
<network_arch/dense_res_block_3/output/kernel/Regularizer/SumSumCnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Square:y:0Gnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_3/output/kernel/Regularizer/SumΕ
>network_arch/dense_res_block_3/output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22@
>network_arch/dense_res_block_3/output/kernel/Regularizer/mul/x΄
<network_arch/dense_res_block_3/output/kernel/Regularizer/mulMulGnetwork_arch/dense_res_block_3/output/kernel/Regularizer/mul/x:output:0Enetwork_arch/dense_res_block_3/output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_3/output/kernel/Regularizer/mulΕ
>network_arch/dense_res_block_3/output/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>network_arch/dense_res_block_3/output/kernel/Regularizer/add/x±
<network_arch/dense_res_block_3/output/kernel/Regularizer/addAddV2Gnetwork_arch/dense_res_block_3/output/kernel/Regularizer/add/x:output:0@network_arch/dense_res_block_3/output/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_3/output/kernel/Regularizer/addα
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpinner_encoder_5929759*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpν
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
1network_arch/inner_encoder/kernel/Regularizer/Sum―
3network_arch/inner_encoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+225
3network_arch/inner_encoder/kernel/Regularizer/mul/x
1network_arch/inner_encoder/kernel/Regularizer/mulMul<network_arch/inner_encoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_encoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/mul―
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
1network_arch/inner_encoder/kernel/Regularizer/addα
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpinner_decoder_5929787*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpν
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
1network_arch/inner_decoder/kernel/Regularizer/Sum―
3network_arch/inner_decoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+225
3network_arch/inner_decoder/kernel/Regularizer/mul/x
1network_arch/inner_decoder/kernel/Regularizer/mulMul<network_arch/inner_decoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_decoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/mul―
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
1network_arch/inner_decoder/kernel/Regularizer/addΤ'
Identity_100IdentityReshape_4:output:0*^dense_res_block_2/StatefulPartitionedCall,^dense_res_block_2_1/StatefulPartitionedCall,^dense_res_block_2_2/StatefulPartitionedCall,^dense_res_block_2_3/StatefulPartitionedCall*^dense_res_block_3/StatefulPartitionedCall,^dense_res_block_3_1/StatefulPartitionedCall-^dense_res_block_3_10/StatefulPartitionedCall-^dense_res_block_3_11/StatefulPartitionedCall-^dense_res_block_3_12/StatefulPartitionedCall-^dense_res_block_3_13/StatefulPartitionedCall-^dense_res_block_3_14/StatefulPartitionedCall-^dense_res_block_3_15/StatefulPartitionedCall-^dense_res_block_3_16/StatefulPartitionedCall-^dense_res_block_3_17/StatefulPartitionedCall-^dense_res_block_3_18/StatefulPartitionedCall-^dense_res_block_3_19/StatefulPartitionedCall,^dense_res_block_3_2/StatefulPartitionedCall-^dense_res_block_3_20/StatefulPartitionedCall-^dense_res_block_3_21/StatefulPartitionedCall-^dense_res_block_3_22/StatefulPartitionedCall-^dense_res_block_3_23/StatefulPartitionedCall-^dense_res_block_3_24/StatefulPartitionedCall-^dense_res_block_3_25/StatefulPartitionedCall-^dense_res_block_3_26/StatefulPartitionedCall-^dense_res_block_3_27/StatefulPartitionedCall-^dense_res_block_3_28/StatefulPartitionedCall-^dense_res_block_3_29/StatefulPartitionedCall,^dense_res_block_3_3/StatefulPartitionedCall-^dense_res_block_3_30/StatefulPartitionedCall-^dense_res_block_3_31/StatefulPartitionedCall-^dense_res_block_3_32/StatefulPartitionedCall-^dense_res_block_3_33/StatefulPartitionedCall-^dense_res_block_3_34/StatefulPartitionedCall-^dense_res_block_3_35/StatefulPartitionedCall-^dense_res_block_3_36/StatefulPartitionedCall-^dense_res_block_3_37/StatefulPartitionedCall-^dense_res_block_3_38/StatefulPartitionedCall-^dense_res_block_3_39/StatefulPartitionedCall,^dense_res_block_3_4/StatefulPartitionedCall-^dense_res_block_3_40/StatefulPartitionedCall-^dense_res_block_3_41/StatefulPartitionedCall-^dense_res_block_3_42/StatefulPartitionedCall-^dense_res_block_3_43/StatefulPartitionedCall-^dense_res_block_3_44/StatefulPartitionedCall-^dense_res_block_3_45/StatefulPartitionedCall-^dense_res_block_3_46/StatefulPartitionedCall-^dense_res_block_3_47/StatefulPartitionedCall-^dense_res_block_3_48/StatefulPartitionedCall-^dense_res_block_3_49/StatefulPartitionedCall,^dense_res_block_3_5/StatefulPartitionedCall-^dense_res_block_3_50/StatefulPartitionedCall-^dense_res_block_3_51/StatefulPartitionedCall,^dense_res_block_3_6/StatefulPartitionedCall,^dense_res_block_3_7/StatefulPartitionedCall,^dense_res_block_3_8/StatefulPartitionedCall,^dense_res_block_3_9/StatefulPartitionedCall&^inner_decoder/StatefulPartitionedCall(^inner_decoder_1/StatefulPartitionedCall)^inner_decoder_10/StatefulPartitionedCall)^inner_decoder_11/StatefulPartitionedCall)^inner_decoder_12/StatefulPartitionedCall)^inner_decoder_13/StatefulPartitionedCall)^inner_decoder_14/StatefulPartitionedCall)^inner_decoder_15/StatefulPartitionedCall)^inner_decoder_16/StatefulPartitionedCall)^inner_decoder_17/StatefulPartitionedCall)^inner_decoder_18/StatefulPartitionedCall)^inner_decoder_19/StatefulPartitionedCall(^inner_decoder_2/StatefulPartitionedCall)^inner_decoder_20/StatefulPartitionedCall)^inner_decoder_21/StatefulPartitionedCall)^inner_decoder_22/StatefulPartitionedCall)^inner_decoder_23/StatefulPartitionedCall)^inner_decoder_24/StatefulPartitionedCall)^inner_decoder_25/StatefulPartitionedCall)^inner_decoder_26/StatefulPartitionedCall)^inner_decoder_27/StatefulPartitionedCall)^inner_decoder_28/StatefulPartitionedCall)^inner_decoder_29/StatefulPartitionedCall(^inner_decoder_3/StatefulPartitionedCall)^inner_decoder_30/StatefulPartitionedCall)^inner_decoder_31/StatefulPartitionedCall)^inner_decoder_32/StatefulPartitionedCall)^inner_decoder_33/StatefulPartitionedCall)^inner_decoder_34/StatefulPartitionedCall)^inner_decoder_35/StatefulPartitionedCall)^inner_decoder_36/StatefulPartitionedCall)^inner_decoder_37/StatefulPartitionedCall)^inner_decoder_38/StatefulPartitionedCall)^inner_decoder_39/StatefulPartitionedCall(^inner_decoder_4/StatefulPartitionedCall)^inner_decoder_40/StatefulPartitionedCall)^inner_decoder_41/StatefulPartitionedCall)^inner_decoder_42/StatefulPartitionedCall)^inner_decoder_43/StatefulPartitionedCall)^inner_decoder_44/StatefulPartitionedCall)^inner_decoder_45/StatefulPartitionedCall)^inner_decoder_46/StatefulPartitionedCall)^inner_decoder_47/StatefulPartitionedCall)^inner_decoder_48/StatefulPartitionedCall)^inner_decoder_49/StatefulPartitionedCall(^inner_decoder_5/StatefulPartitionedCall)^inner_decoder_50/StatefulPartitionedCall(^inner_decoder_6/StatefulPartitionedCall(^inner_decoder_7/StatefulPartitionedCall(^inner_decoder_8/StatefulPartitionedCall(^inner_decoder_9/StatefulPartitionedCall&^inner_encoder/StatefulPartitionedCall(^inner_encoder_1/StatefulPartitionedCall(^inner_encoder_2/StatefulPartitionedCall(^inner_encoder_3/StatefulPartitionedCall*
T0*,
_output_shapes
:?????????32
Identity_100Τ'
Identity_101IdentityReshape_5:output:0*^dense_res_block_2/StatefulPartitionedCall,^dense_res_block_2_1/StatefulPartitionedCall,^dense_res_block_2_2/StatefulPartitionedCall,^dense_res_block_2_3/StatefulPartitionedCall*^dense_res_block_3/StatefulPartitionedCall,^dense_res_block_3_1/StatefulPartitionedCall-^dense_res_block_3_10/StatefulPartitionedCall-^dense_res_block_3_11/StatefulPartitionedCall-^dense_res_block_3_12/StatefulPartitionedCall-^dense_res_block_3_13/StatefulPartitionedCall-^dense_res_block_3_14/StatefulPartitionedCall-^dense_res_block_3_15/StatefulPartitionedCall-^dense_res_block_3_16/StatefulPartitionedCall-^dense_res_block_3_17/StatefulPartitionedCall-^dense_res_block_3_18/StatefulPartitionedCall-^dense_res_block_3_19/StatefulPartitionedCall,^dense_res_block_3_2/StatefulPartitionedCall-^dense_res_block_3_20/StatefulPartitionedCall-^dense_res_block_3_21/StatefulPartitionedCall-^dense_res_block_3_22/StatefulPartitionedCall-^dense_res_block_3_23/StatefulPartitionedCall-^dense_res_block_3_24/StatefulPartitionedCall-^dense_res_block_3_25/StatefulPartitionedCall-^dense_res_block_3_26/StatefulPartitionedCall-^dense_res_block_3_27/StatefulPartitionedCall-^dense_res_block_3_28/StatefulPartitionedCall-^dense_res_block_3_29/StatefulPartitionedCall,^dense_res_block_3_3/StatefulPartitionedCall-^dense_res_block_3_30/StatefulPartitionedCall-^dense_res_block_3_31/StatefulPartitionedCall-^dense_res_block_3_32/StatefulPartitionedCall-^dense_res_block_3_33/StatefulPartitionedCall-^dense_res_block_3_34/StatefulPartitionedCall-^dense_res_block_3_35/StatefulPartitionedCall-^dense_res_block_3_36/StatefulPartitionedCall-^dense_res_block_3_37/StatefulPartitionedCall-^dense_res_block_3_38/StatefulPartitionedCall-^dense_res_block_3_39/StatefulPartitionedCall,^dense_res_block_3_4/StatefulPartitionedCall-^dense_res_block_3_40/StatefulPartitionedCall-^dense_res_block_3_41/StatefulPartitionedCall-^dense_res_block_3_42/StatefulPartitionedCall-^dense_res_block_3_43/StatefulPartitionedCall-^dense_res_block_3_44/StatefulPartitionedCall-^dense_res_block_3_45/StatefulPartitionedCall-^dense_res_block_3_46/StatefulPartitionedCall-^dense_res_block_3_47/StatefulPartitionedCall-^dense_res_block_3_48/StatefulPartitionedCall-^dense_res_block_3_49/StatefulPartitionedCall,^dense_res_block_3_5/StatefulPartitionedCall-^dense_res_block_3_50/StatefulPartitionedCall-^dense_res_block_3_51/StatefulPartitionedCall,^dense_res_block_3_6/StatefulPartitionedCall,^dense_res_block_3_7/StatefulPartitionedCall,^dense_res_block_3_8/StatefulPartitionedCall,^dense_res_block_3_9/StatefulPartitionedCall&^inner_decoder/StatefulPartitionedCall(^inner_decoder_1/StatefulPartitionedCall)^inner_decoder_10/StatefulPartitionedCall)^inner_decoder_11/StatefulPartitionedCall)^inner_decoder_12/StatefulPartitionedCall)^inner_decoder_13/StatefulPartitionedCall)^inner_decoder_14/StatefulPartitionedCall)^inner_decoder_15/StatefulPartitionedCall)^inner_decoder_16/StatefulPartitionedCall)^inner_decoder_17/StatefulPartitionedCall)^inner_decoder_18/StatefulPartitionedCall)^inner_decoder_19/StatefulPartitionedCall(^inner_decoder_2/StatefulPartitionedCall)^inner_decoder_20/StatefulPartitionedCall)^inner_decoder_21/StatefulPartitionedCall)^inner_decoder_22/StatefulPartitionedCall)^inner_decoder_23/StatefulPartitionedCall)^inner_decoder_24/StatefulPartitionedCall)^inner_decoder_25/StatefulPartitionedCall)^inner_decoder_26/StatefulPartitionedCall)^inner_decoder_27/StatefulPartitionedCall)^inner_decoder_28/StatefulPartitionedCall)^inner_decoder_29/StatefulPartitionedCall(^inner_decoder_3/StatefulPartitionedCall)^inner_decoder_30/StatefulPartitionedCall)^inner_decoder_31/StatefulPartitionedCall)^inner_decoder_32/StatefulPartitionedCall)^inner_decoder_33/StatefulPartitionedCall)^inner_decoder_34/StatefulPartitionedCall)^inner_decoder_35/StatefulPartitionedCall)^inner_decoder_36/StatefulPartitionedCall)^inner_decoder_37/StatefulPartitionedCall)^inner_decoder_38/StatefulPartitionedCall)^inner_decoder_39/StatefulPartitionedCall(^inner_decoder_4/StatefulPartitionedCall)^inner_decoder_40/StatefulPartitionedCall)^inner_decoder_41/StatefulPartitionedCall)^inner_decoder_42/StatefulPartitionedCall)^inner_decoder_43/StatefulPartitionedCall)^inner_decoder_44/StatefulPartitionedCall)^inner_decoder_45/StatefulPartitionedCall)^inner_decoder_46/StatefulPartitionedCall)^inner_decoder_47/StatefulPartitionedCall)^inner_decoder_48/StatefulPartitionedCall)^inner_decoder_49/StatefulPartitionedCall(^inner_decoder_5/StatefulPartitionedCall)^inner_decoder_50/StatefulPartitionedCall(^inner_decoder_6/StatefulPartitionedCall(^inner_decoder_7/StatefulPartitionedCall(^inner_decoder_8/StatefulPartitionedCall(^inner_decoder_9/StatefulPartitionedCall&^inner_encoder/StatefulPartitionedCall(^inner_encoder_1/StatefulPartitionedCall(^inner_encoder_2/StatefulPartitionedCall(^inner_encoder_3/StatefulPartitionedCall*
T0*,
_output_shapes
:?????????32
Identity_101Σ'
Identity_102Identityconcat_2:output:0*^dense_res_block_2/StatefulPartitionedCall,^dense_res_block_2_1/StatefulPartitionedCall,^dense_res_block_2_2/StatefulPartitionedCall,^dense_res_block_2_3/StatefulPartitionedCall*^dense_res_block_3/StatefulPartitionedCall,^dense_res_block_3_1/StatefulPartitionedCall-^dense_res_block_3_10/StatefulPartitionedCall-^dense_res_block_3_11/StatefulPartitionedCall-^dense_res_block_3_12/StatefulPartitionedCall-^dense_res_block_3_13/StatefulPartitionedCall-^dense_res_block_3_14/StatefulPartitionedCall-^dense_res_block_3_15/StatefulPartitionedCall-^dense_res_block_3_16/StatefulPartitionedCall-^dense_res_block_3_17/StatefulPartitionedCall-^dense_res_block_3_18/StatefulPartitionedCall-^dense_res_block_3_19/StatefulPartitionedCall,^dense_res_block_3_2/StatefulPartitionedCall-^dense_res_block_3_20/StatefulPartitionedCall-^dense_res_block_3_21/StatefulPartitionedCall-^dense_res_block_3_22/StatefulPartitionedCall-^dense_res_block_3_23/StatefulPartitionedCall-^dense_res_block_3_24/StatefulPartitionedCall-^dense_res_block_3_25/StatefulPartitionedCall-^dense_res_block_3_26/StatefulPartitionedCall-^dense_res_block_3_27/StatefulPartitionedCall-^dense_res_block_3_28/StatefulPartitionedCall-^dense_res_block_3_29/StatefulPartitionedCall,^dense_res_block_3_3/StatefulPartitionedCall-^dense_res_block_3_30/StatefulPartitionedCall-^dense_res_block_3_31/StatefulPartitionedCall-^dense_res_block_3_32/StatefulPartitionedCall-^dense_res_block_3_33/StatefulPartitionedCall-^dense_res_block_3_34/StatefulPartitionedCall-^dense_res_block_3_35/StatefulPartitionedCall-^dense_res_block_3_36/StatefulPartitionedCall-^dense_res_block_3_37/StatefulPartitionedCall-^dense_res_block_3_38/StatefulPartitionedCall-^dense_res_block_3_39/StatefulPartitionedCall,^dense_res_block_3_4/StatefulPartitionedCall-^dense_res_block_3_40/StatefulPartitionedCall-^dense_res_block_3_41/StatefulPartitionedCall-^dense_res_block_3_42/StatefulPartitionedCall-^dense_res_block_3_43/StatefulPartitionedCall-^dense_res_block_3_44/StatefulPartitionedCall-^dense_res_block_3_45/StatefulPartitionedCall-^dense_res_block_3_46/StatefulPartitionedCall-^dense_res_block_3_47/StatefulPartitionedCall-^dense_res_block_3_48/StatefulPartitionedCall-^dense_res_block_3_49/StatefulPartitionedCall,^dense_res_block_3_5/StatefulPartitionedCall-^dense_res_block_3_50/StatefulPartitionedCall-^dense_res_block_3_51/StatefulPartitionedCall,^dense_res_block_3_6/StatefulPartitionedCall,^dense_res_block_3_7/StatefulPartitionedCall,^dense_res_block_3_8/StatefulPartitionedCall,^dense_res_block_3_9/StatefulPartitionedCall&^inner_decoder/StatefulPartitionedCall(^inner_decoder_1/StatefulPartitionedCall)^inner_decoder_10/StatefulPartitionedCall)^inner_decoder_11/StatefulPartitionedCall)^inner_decoder_12/StatefulPartitionedCall)^inner_decoder_13/StatefulPartitionedCall)^inner_decoder_14/StatefulPartitionedCall)^inner_decoder_15/StatefulPartitionedCall)^inner_decoder_16/StatefulPartitionedCall)^inner_decoder_17/StatefulPartitionedCall)^inner_decoder_18/StatefulPartitionedCall)^inner_decoder_19/StatefulPartitionedCall(^inner_decoder_2/StatefulPartitionedCall)^inner_decoder_20/StatefulPartitionedCall)^inner_decoder_21/StatefulPartitionedCall)^inner_decoder_22/StatefulPartitionedCall)^inner_decoder_23/StatefulPartitionedCall)^inner_decoder_24/StatefulPartitionedCall)^inner_decoder_25/StatefulPartitionedCall)^inner_decoder_26/StatefulPartitionedCall)^inner_decoder_27/StatefulPartitionedCall)^inner_decoder_28/StatefulPartitionedCall)^inner_decoder_29/StatefulPartitionedCall(^inner_decoder_3/StatefulPartitionedCall)^inner_decoder_30/StatefulPartitionedCall)^inner_decoder_31/StatefulPartitionedCall)^inner_decoder_32/StatefulPartitionedCall)^inner_decoder_33/StatefulPartitionedCall)^inner_decoder_34/StatefulPartitionedCall)^inner_decoder_35/StatefulPartitionedCall)^inner_decoder_36/StatefulPartitionedCall)^inner_decoder_37/StatefulPartitionedCall)^inner_decoder_38/StatefulPartitionedCall)^inner_decoder_39/StatefulPartitionedCall(^inner_decoder_4/StatefulPartitionedCall)^inner_decoder_40/StatefulPartitionedCall)^inner_decoder_41/StatefulPartitionedCall)^inner_decoder_42/StatefulPartitionedCall)^inner_decoder_43/StatefulPartitionedCall)^inner_decoder_44/StatefulPartitionedCall)^inner_decoder_45/StatefulPartitionedCall)^inner_decoder_46/StatefulPartitionedCall)^inner_decoder_47/StatefulPartitionedCall)^inner_decoder_48/StatefulPartitionedCall)^inner_decoder_49/StatefulPartitionedCall(^inner_decoder_5/StatefulPartitionedCall)^inner_decoder_50/StatefulPartitionedCall(^inner_decoder_6/StatefulPartitionedCall(^inner_decoder_7/StatefulPartitionedCall(^inner_decoder_8/StatefulPartitionedCall(^inner_decoder_9/StatefulPartitionedCall&^inner_encoder/StatefulPartitionedCall(^inner_encoder_1/StatefulPartitionedCall(^inner_encoder_2/StatefulPartitionedCall(^inner_encoder_3/StatefulPartitionedCall*
T0*,
_output_shapes
:?????????22
Identity_102"%
identity_100Identity_100:output:0"%
identity_101Identity_101:output:0"%
identity_102Identity_102:output:0*
_input_shapesv
t:?????????3:::::::::::::::::::::::2V
)dense_res_block_2/StatefulPartitionedCall)dense_res_block_2/StatefulPartitionedCall2Z
+dense_res_block_2_1/StatefulPartitionedCall+dense_res_block_2_1/StatefulPartitionedCall2Z
+dense_res_block_2_2/StatefulPartitionedCall+dense_res_block_2_2/StatefulPartitionedCall2Z
+dense_res_block_2_3/StatefulPartitionedCall+dense_res_block_2_3/StatefulPartitionedCall2V
)dense_res_block_3/StatefulPartitionedCall)dense_res_block_3/StatefulPartitionedCall2Z
+dense_res_block_3_1/StatefulPartitionedCall+dense_res_block_3_1/StatefulPartitionedCall2\
,dense_res_block_3_10/StatefulPartitionedCall,dense_res_block_3_10/StatefulPartitionedCall2\
,dense_res_block_3_11/StatefulPartitionedCall,dense_res_block_3_11/StatefulPartitionedCall2\
,dense_res_block_3_12/StatefulPartitionedCall,dense_res_block_3_12/StatefulPartitionedCall2\
,dense_res_block_3_13/StatefulPartitionedCall,dense_res_block_3_13/StatefulPartitionedCall2\
,dense_res_block_3_14/StatefulPartitionedCall,dense_res_block_3_14/StatefulPartitionedCall2\
,dense_res_block_3_15/StatefulPartitionedCall,dense_res_block_3_15/StatefulPartitionedCall2\
,dense_res_block_3_16/StatefulPartitionedCall,dense_res_block_3_16/StatefulPartitionedCall2\
,dense_res_block_3_17/StatefulPartitionedCall,dense_res_block_3_17/StatefulPartitionedCall2\
,dense_res_block_3_18/StatefulPartitionedCall,dense_res_block_3_18/StatefulPartitionedCall2\
,dense_res_block_3_19/StatefulPartitionedCall,dense_res_block_3_19/StatefulPartitionedCall2Z
+dense_res_block_3_2/StatefulPartitionedCall+dense_res_block_3_2/StatefulPartitionedCall2\
,dense_res_block_3_20/StatefulPartitionedCall,dense_res_block_3_20/StatefulPartitionedCall2\
,dense_res_block_3_21/StatefulPartitionedCall,dense_res_block_3_21/StatefulPartitionedCall2\
,dense_res_block_3_22/StatefulPartitionedCall,dense_res_block_3_22/StatefulPartitionedCall2\
,dense_res_block_3_23/StatefulPartitionedCall,dense_res_block_3_23/StatefulPartitionedCall2\
,dense_res_block_3_24/StatefulPartitionedCall,dense_res_block_3_24/StatefulPartitionedCall2\
,dense_res_block_3_25/StatefulPartitionedCall,dense_res_block_3_25/StatefulPartitionedCall2\
,dense_res_block_3_26/StatefulPartitionedCall,dense_res_block_3_26/StatefulPartitionedCall2\
,dense_res_block_3_27/StatefulPartitionedCall,dense_res_block_3_27/StatefulPartitionedCall2\
,dense_res_block_3_28/StatefulPartitionedCall,dense_res_block_3_28/StatefulPartitionedCall2\
,dense_res_block_3_29/StatefulPartitionedCall,dense_res_block_3_29/StatefulPartitionedCall2Z
+dense_res_block_3_3/StatefulPartitionedCall+dense_res_block_3_3/StatefulPartitionedCall2\
,dense_res_block_3_30/StatefulPartitionedCall,dense_res_block_3_30/StatefulPartitionedCall2\
,dense_res_block_3_31/StatefulPartitionedCall,dense_res_block_3_31/StatefulPartitionedCall2\
,dense_res_block_3_32/StatefulPartitionedCall,dense_res_block_3_32/StatefulPartitionedCall2\
,dense_res_block_3_33/StatefulPartitionedCall,dense_res_block_3_33/StatefulPartitionedCall2\
,dense_res_block_3_34/StatefulPartitionedCall,dense_res_block_3_34/StatefulPartitionedCall2\
,dense_res_block_3_35/StatefulPartitionedCall,dense_res_block_3_35/StatefulPartitionedCall2\
,dense_res_block_3_36/StatefulPartitionedCall,dense_res_block_3_36/StatefulPartitionedCall2\
,dense_res_block_3_37/StatefulPartitionedCall,dense_res_block_3_37/StatefulPartitionedCall2\
,dense_res_block_3_38/StatefulPartitionedCall,dense_res_block_3_38/StatefulPartitionedCall2\
,dense_res_block_3_39/StatefulPartitionedCall,dense_res_block_3_39/StatefulPartitionedCall2Z
+dense_res_block_3_4/StatefulPartitionedCall+dense_res_block_3_4/StatefulPartitionedCall2\
,dense_res_block_3_40/StatefulPartitionedCall,dense_res_block_3_40/StatefulPartitionedCall2\
,dense_res_block_3_41/StatefulPartitionedCall,dense_res_block_3_41/StatefulPartitionedCall2\
,dense_res_block_3_42/StatefulPartitionedCall,dense_res_block_3_42/StatefulPartitionedCall2\
,dense_res_block_3_43/StatefulPartitionedCall,dense_res_block_3_43/StatefulPartitionedCall2\
,dense_res_block_3_44/StatefulPartitionedCall,dense_res_block_3_44/StatefulPartitionedCall2\
,dense_res_block_3_45/StatefulPartitionedCall,dense_res_block_3_45/StatefulPartitionedCall2\
,dense_res_block_3_46/StatefulPartitionedCall,dense_res_block_3_46/StatefulPartitionedCall2\
,dense_res_block_3_47/StatefulPartitionedCall,dense_res_block_3_47/StatefulPartitionedCall2\
,dense_res_block_3_48/StatefulPartitionedCall,dense_res_block_3_48/StatefulPartitionedCall2\
,dense_res_block_3_49/StatefulPartitionedCall,dense_res_block_3_49/StatefulPartitionedCall2Z
+dense_res_block_3_5/StatefulPartitionedCall+dense_res_block_3_5/StatefulPartitionedCall2\
,dense_res_block_3_50/StatefulPartitionedCall,dense_res_block_3_50/StatefulPartitionedCall2\
,dense_res_block_3_51/StatefulPartitionedCall,dense_res_block_3_51/StatefulPartitionedCall2Z
+dense_res_block_3_6/StatefulPartitionedCall+dense_res_block_3_6/StatefulPartitionedCall2Z
+dense_res_block_3_7/StatefulPartitionedCall+dense_res_block_3_7/StatefulPartitionedCall2Z
+dense_res_block_3_8/StatefulPartitionedCall+dense_res_block_3_8/StatefulPartitionedCall2Z
+dense_res_block_3_9/StatefulPartitionedCall+dense_res_block_3_9/StatefulPartitionedCall2N
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
:?????????3
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
: 
Τ

K__inference_inner_encoder_layer_call_and_return_conditional_losses_16856549

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
:?????????2
MatMulκ
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpν
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
1network_arch/inner_encoder/kernel/Regularizer/Sum―
3network_arch/inner_encoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+225
3network_arch/inner_encoder/kernel/Regularizer/mul/x
1network_arch/inner_encoder/kernel/Regularizer/mulMul<network_arch/inner_encoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_encoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/mul―
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
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
l
Σ
$__inference__traced_restore_16863135
file_prefix
assignvariableop_variable8
4assignvariableop_1_network_arch_inner_encoder_kernel8
4assignvariableop_2_network_arch_inner_decoder_kernelD
@assignvariableop_3_network_arch_dense_res_block_2_hidden0_kernelB
>assignvariableop_4_network_arch_dense_res_block_2_hidden0_biasD
@assignvariableop_5_network_arch_dense_res_block_2_hidden1_kernelB
>assignvariableop_6_network_arch_dense_res_block_2_hidden1_biasD
@assignvariableop_7_network_arch_dense_res_block_2_hidden2_kernelB
>assignvariableop_8_network_arch_dense_res_block_2_hidden2_biasD
@assignvariableop_9_network_arch_dense_res_block_2_hidden3_kernelC
?assignvariableop_10_network_arch_dense_res_block_2_hidden3_biasD
@assignvariableop_11_network_arch_dense_res_block_2_output_kernelB
>assignvariableop_12_network_arch_dense_res_block_2_output_biasE
Aassignvariableop_13_network_arch_dense_res_block_3_hidden0_kernelC
?assignvariableop_14_network_arch_dense_res_block_3_hidden0_biasE
Aassignvariableop_15_network_arch_dense_res_block_3_hidden1_kernelC
?assignvariableop_16_network_arch_dense_res_block_3_hidden1_biasE
Aassignvariableop_17_network_arch_dense_res_block_3_hidden2_kernelC
?assignvariableop_18_network_arch_dense_res_block_3_hidden2_biasE
Aassignvariableop_19_network_arch_dense_res_block_3_hidden3_kernelC
?assignvariableop_20_network_arch_dense_res_block_3_hidden3_biasD
@assignvariableop_21_network_arch_dense_res_block_3_output_kernelB
>assignvariableop_22_network_arch_dense_res_block_3_output_bias
identity_24’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9’	RestoreV2’RestoreV2_1³
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ώ
value΅B²BL/.ATTRIBUTES/VARIABLE_VALUEB/inner_encoder/kernel/.ATTRIBUTES/VARIABLE_VALUEB/inner_decoder/kernel/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesΌ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
22
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

Identity_1ͺ
AssignVariableOp_1AssignVariableOp4assignvariableop_1_network_arch_inner_encoder_kernelIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2ͺ
AssignVariableOp_2AssignVariableOp4assignvariableop_2_network_arch_inner_decoder_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Ά
AssignVariableOp_3AssignVariableOp@assignvariableop_3_network_arch_dense_res_block_2_hidden0_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4΄
AssignVariableOp_4AssignVariableOp>assignvariableop_4_network_arch_dense_res_block_2_hidden0_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Ά
AssignVariableOp_5AssignVariableOp@assignvariableop_5_network_arch_dense_res_block_2_hidden1_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6΄
AssignVariableOp_6AssignVariableOp>assignvariableop_6_network_arch_dense_res_block_2_hidden1_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ά
AssignVariableOp_7AssignVariableOp@assignvariableop_7_network_arch_dense_res_block_2_hidden2_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8΄
AssignVariableOp_8AssignVariableOp>assignvariableop_8_network_arch_dense_res_block_2_hidden2_biasIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Ά
AssignVariableOp_9AssignVariableOp@assignvariableop_9_network_arch_dense_res_block_2_hidden3_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10Έ
AssignVariableOp_10AssignVariableOp?assignvariableop_10_network_arch_dense_res_block_2_hidden3_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11Ή
AssignVariableOp_11AssignVariableOp@assignvariableop_11_network_arch_dense_res_block_2_output_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12·
AssignVariableOp_12AssignVariableOp>assignvariableop_12_network_arch_dense_res_block_2_output_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Ί
AssignVariableOp_13AssignVariableOpAassignvariableop_13_network_arch_dense_res_block_3_hidden0_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Έ
AssignVariableOp_14AssignVariableOp?assignvariableop_14_network_arch_dense_res_block_3_hidden0_biasIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Ί
AssignVariableOp_15AssignVariableOpAassignvariableop_15_network_arch_dense_res_block_3_hidden1_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16Έ
AssignVariableOp_16AssignVariableOp?assignvariableop_16_network_arch_dense_res_block_3_hidden1_biasIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Ί
AssignVariableOp_17AssignVariableOpAassignvariableop_17_network_arch_dense_res_block_3_hidden2_kernelIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Έ
AssignVariableOp_18AssignVariableOp?assignvariableop_18_network_arch_dense_res_block_3_hidden2_biasIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Ί
AssignVariableOp_19AssignVariableOpAassignvariableop_19_network_arch_dense_res_block_3_hidden3_kernelIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Έ
AssignVariableOp_20AssignVariableOp?assignvariableop_20_network_arch_dense_res_block_3_hidden3_biasIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21Ή
AssignVariableOp_21AssignVariableOp@assignvariableop_21_network_arch_dense_res_block_3_output_kernelIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22·
AssignVariableOp_22AssignVariableOp>assignvariableop_22_network_arch_dense_res_block_3_output_biasIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22¨
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
RestoreV2_1/shape_and_slicesΔ
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
NoOpΨ
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23ε
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_24"#
identity_24Identity_24:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2$
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
AssignVariableOp_22AssignVariableOp_222(
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
: 
ΰ
h
__inference_loss_fn_4_168627855
1kernel_regularizer_square_readvariableop_resource
identityΘ
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
 *wΜ+22
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
ΰ
h
__inference_loss_fn_8_168628375
1kernel_regularizer_square_readvariableop_resource
identityΘ
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
 *wΜ+22
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
ΒD
±
!__inference__traced_save_16863054
file_prefix'
#savev2_variable_read_readvariableop@
<savev2_network_arch_inner_encoder_kernel_read_readvariableop@
<savev2_network_arch_inner_decoder_kernel_read_readvariableopL
Hsavev2_network_arch_dense_res_block_2_hidden0_kernel_read_readvariableopJ
Fsavev2_network_arch_dense_res_block_2_hidden0_bias_read_readvariableopL
Hsavev2_network_arch_dense_res_block_2_hidden1_kernel_read_readvariableopJ
Fsavev2_network_arch_dense_res_block_2_hidden1_bias_read_readvariableopL
Hsavev2_network_arch_dense_res_block_2_hidden2_kernel_read_readvariableopJ
Fsavev2_network_arch_dense_res_block_2_hidden2_bias_read_readvariableopL
Hsavev2_network_arch_dense_res_block_2_hidden3_kernel_read_readvariableopJ
Fsavev2_network_arch_dense_res_block_2_hidden3_bias_read_readvariableopK
Gsavev2_network_arch_dense_res_block_2_output_kernel_read_readvariableopI
Esavev2_network_arch_dense_res_block_2_output_bias_read_readvariableopL
Hsavev2_network_arch_dense_res_block_3_hidden0_kernel_read_readvariableopJ
Fsavev2_network_arch_dense_res_block_3_hidden0_bias_read_readvariableopL
Hsavev2_network_arch_dense_res_block_3_hidden1_kernel_read_readvariableopJ
Fsavev2_network_arch_dense_res_block_3_hidden1_bias_read_readvariableopL
Hsavev2_network_arch_dense_res_block_3_hidden2_kernel_read_readvariableopJ
Fsavev2_network_arch_dense_res_block_3_hidden2_bias_read_readvariableopL
Hsavev2_network_arch_dense_res_block_3_hidden3_kernel_read_readvariableopJ
Fsavev2_network_arch_dense_res_block_3_hidden3_bias_read_readvariableopK
Gsavev2_network_arch_dense_res_block_3_output_kernel_read_readvariableopI
Esavev2_network_arch_dense_res_block_3_output_bias_read_readvariableop
savev2_1_const

identity_1’MergeV2Checkpoints’SaveV2’SaveV2_1
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
value3B1 B+_temp_759afae2b22b4b8aa82c7e77620218ff/part2	
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
ShardedFilename­
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ώ
value΅B²BL/.ATTRIBUTES/VARIABLE_VALUEB/inner_encoder/kernel/.ATTRIBUTES/VARIABLE_VALUEB/inner_decoder/kernel/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesΆ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop<savev2_network_arch_inner_encoder_kernel_read_readvariableop<savev2_network_arch_inner_decoder_kernel_read_readvariableopHsavev2_network_arch_dense_res_block_2_hidden0_kernel_read_readvariableopFsavev2_network_arch_dense_res_block_2_hidden0_bias_read_readvariableopHsavev2_network_arch_dense_res_block_2_hidden1_kernel_read_readvariableopFsavev2_network_arch_dense_res_block_2_hidden1_bias_read_readvariableopHsavev2_network_arch_dense_res_block_2_hidden2_kernel_read_readvariableopFsavev2_network_arch_dense_res_block_2_hidden2_bias_read_readvariableopHsavev2_network_arch_dense_res_block_2_hidden3_kernel_read_readvariableopFsavev2_network_arch_dense_res_block_2_hidden3_bias_read_readvariableopGsavev2_network_arch_dense_res_block_2_output_kernel_read_readvariableopEsavev2_network_arch_dense_res_block_2_output_bias_read_readvariableopHsavev2_network_arch_dense_res_block_3_hidden0_kernel_read_readvariableopFsavev2_network_arch_dense_res_block_3_hidden0_bias_read_readvariableopHsavev2_network_arch_dense_res_block_3_hidden1_kernel_read_readvariableopFsavev2_network_arch_dense_res_block_3_hidden1_bias_read_readvariableopHsavev2_network_arch_dense_res_block_3_hidden2_kernel_read_readvariableopFsavev2_network_arch_dense_res_block_3_hidden2_bias_read_readvariableopHsavev2_network_arch_dense_res_block_3_hidden3_kernel_read_readvariableopFsavev2_network_arch_dense_res_block_3_hidden3_bias_read_readvariableopGsavev2_network_arch_dense_res_block_3_output_kernel_read_readvariableopEsavev2_network_arch_dense_res_block_3_output_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *%
dtypes
22
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
ShardedFilename_1’
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
SaveV2_1/shape_and_slicesΟ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1γ
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

identity_1Identity_1:output:0*χ
_input_shapesε
β: ::	:	:
::
::
::
::
::
::
::
::
::
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
:	:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!	

_output_shapes	
::&
"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: 

χ
#__inference__wrapped_model_16862511
input_1
network_arch_16862459
network_arch_16862461
network_arch_16862463
network_arch_16862465
network_arch_16862467
network_arch_16862469
network_arch_16862471
network_arch_16862473
network_arch_16862475
network_arch_16862477
network_arch_16862479
network_arch_16862481
network_arch_16862483
network_arch_16862485
network_arch_16862487
network_arch_16862489
network_arch_16862491
network_arch_16862493
network_arch_16862495
network_arch_16862497
network_arch_16862499
network_arch_16862501
network_arch_16862503
identity

identity_1

identity_2’$network_arch/StatefulPartitionedCall²
$network_arch/StatefulPartitionedCallStatefulPartitionedCallinput_1network_arch_16862459network_arch_16862461network_arch_16862463network_arch_16862465network_arch_16862467network_arch_16862469network_arch_16862471network_arch_16862473network_arch_16862475network_arch_16862477network_arch_16862479network_arch_16862481network_arch_16862483network_arch_16862485network_arch_16862487network_arch_16862489network_arch_16862491network_arch_16862493network_arch_16862495network_arch_16862497network_arch_16862499network_arch_16862501network_arch_16862503*#
Tin
2*
Tout
2*\
_output_shapesJ
H:?????????3:?????????3:?????????2*9
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*4
f/R-
+__inference_restored_function_body_168624582&
$network_arch/StatefulPartitionedCall­
IdentityIdentity-network_arch/StatefulPartitionedCall:output:0%^network_arch/StatefulPartitionedCall*
T0*,
_output_shapes
:?????????32

Identity±

Identity_1Identity-network_arch/StatefulPartitionedCall:output:1%^network_arch/StatefulPartitionedCall*
T0*,
_output_shapes
:?????????32

Identity_1±

Identity_2Identity-network_arch/StatefulPartitionedCall:output:2%^network_arch/StatefulPartitionedCall*
T0*,
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesv
t:?????????3:::::::::::::::::::::::2L
$network_arch/StatefulPartitionedCall$network_arch/StatefulPartitionedCall:U Q
,
_output_shapes
:?????????3
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
: 


4__inference_dense_res_block_3_layer_call_fn_16856648

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
	unknown_8
identity’StatefulPartitionedCallΖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_168566332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
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
: 
Τ

K__inference_inner_decoder_layer_call_and_return_conditional_losses_16857265

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
:?????????2
MatMulκ
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpν
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
1network_arch/inner_decoder/kernel/Regularizer/Sum―
3network_arch/inner_decoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+225
3network_arch/inner_decoder/kernel/Regularizer/mul/x
1network_arch/inner_decoder/kernel/Regularizer/mulMul<network_arch/inner_decoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_decoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/mul―
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
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
ΰ
h
__inference_loss_fn_3_168627725
1kernel_regularizer_square_readvariableop_resource
identityΘ
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
 *wΜ+22
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
Τ

K__inference_inner_decoder_layer_call_and_return_conditional_losses_16856813

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
:?????????2
MatMulκ
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpν
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
1network_arch/inner_decoder/kernel/Regularizer/Sum―
3network_arch/inner_decoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+225
3network_arch/inner_decoder/kernel/Regularizer/mul/x
1network_arch/inner_decoder/kernel/Regularizer/mulMul<network_arch/inner_decoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_decoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/mul―
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
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
ΰ
h
__inference_loss_fn_5_168627985
1kernel_regularizer_square_readvariableop_resource
identityΘ
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
 *wΜ+22
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
~
©
O__inference_dense_res_block_2_layer_call_and_return_conditional_losses_16856783

inputs*
&hidden0_matmul_readvariableop_resource+
'hidden0_biasadd_readvariableop_resource*
&hidden1_matmul_readvariableop_resource+
'hidden1_biasadd_readvariableop_resource*
&hidden2_matmul_readvariableop_resource+
'hidden2_biasadd_readvariableop_resource*
&hidden3_matmul_readvariableop_resource+
'hidden3_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity§
hidden0/MatMul/ReadVariableOpReadVariableOp&hidden0_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden0/MatMul/ReadVariableOp
hidden0/MatMulMatMulinputs%hidden0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden0/MatMul₯
hidden0/BiasAdd/ReadVariableOpReadVariableOp'hidden0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden0/BiasAdd/ReadVariableOp’
hidden0/BiasAddBiasAddhidden0/MatMul:product:0&hidden0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden0/BiasAddq
hidden0/ReluReluhidden0/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden0/Relu
hidden0/IdentityIdentityhidden0/Relu:activations:0*
T0*(
_output_shapes
:?????????2
hidden0/Identity§
hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden1/MatMul/ReadVariableOp
hidden1/MatMulMatMulhidden0/Identity:output:0%hidden1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden1/MatMul₯
hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden1/BiasAdd/ReadVariableOp’
hidden1/BiasAddBiasAddhidden1/MatMul:product:0&hidden1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden1/BiasAddq
hidden1/ReluReluhidden1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden1/Relu
hidden1/IdentityIdentityhidden1/Relu:activations:0*
T0*(
_output_shapes
:?????????2
hidden1/Identity§
hidden2/MatMul/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden2/MatMul/ReadVariableOp
hidden2/MatMulMatMulhidden1/Identity:output:0%hidden2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden2/MatMul₯
hidden2/BiasAdd/ReadVariableOpReadVariableOp'hidden2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden2/BiasAdd/ReadVariableOp’
hidden2/BiasAddBiasAddhidden2/MatMul:product:0&hidden2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden2/BiasAddq
hidden2/ReluReluhidden2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden2/Relu
hidden2/IdentityIdentityhidden2/Relu:activations:0*
T0*(
_output_shapes
:?????????2
hidden2/Identity§
hidden3/MatMul/ReadVariableOpReadVariableOp&hidden3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden3/MatMul/ReadVariableOp
hidden3/MatMulMatMulhidden2/Identity:output:0%hidden3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden3/MatMul₯
hidden3/BiasAdd/ReadVariableOpReadVariableOp'hidden3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden3/BiasAdd/ReadVariableOp’
hidden3/BiasAddBiasAddhidden3/MatMul:product:0&hidden3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden3/BiasAddq
hidden3/ReluReluhidden3/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden3/Relu
hidden3/IdentityIdentityhidden3/Relu:activations:0*
T0*(
_output_shapes
:?????????2
hidden3/Identity€
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMulhidden3/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
output/MatMul’
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
output/BiasAddz
output/IdentityIdentityoutput/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
output/Identityh
addAddV2inputsoutput/Identity:output:0*
T0*(
_output_shapes
:?????????2
add
Onetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden0_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_2/hidden0/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_2/hidden0/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/add
Onetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_2/hidden1/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_2/hidden1/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/add
Onetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_2/hidden2/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_2/hidden2/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/add
Onetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_2/hidden3/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_2/hidden3/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/add
Nnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02P
Nnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Square/ReadVariableOp
?network_arch/dense_res_block_2/output/kernel/Regularizer/SquareSquareVnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2A
?network_arch/dense_res_block_2/output/kernel/Regularizer/SquareΡ
>network_arch/dense_res_block_2/output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2@
>network_arch/dense_res_block_2/output/kernel/Regularizer/Const²
<network_arch/dense_res_block_2/output/kernel/Regularizer/SumSumCnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Square:y:0Gnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_2/output/kernel/Regularizer/SumΕ
>network_arch/dense_res_block_2/output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22@
>network_arch/dense_res_block_2/output/kernel/Regularizer/mul/x΄
<network_arch/dense_res_block_2/output/kernel/Regularizer/mulMulGnetwork_arch/dense_res_block_2/output/kernel/Regularizer/mul/x:output:0Enetwork_arch/dense_res_block_2/output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_2/output/kernel/Regularizer/mulΕ
>network_arch/dense_res_block_2/output/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>network_arch/dense_res_block_2/output/kernel/Regularizer/add/x±
<network_arch/dense_res_block_2/output/kernel/Regularizer/addAddV2Gnetwork_arch/dense_res_block_2/output/kernel/Regularizer/add/x:output:0@network_arch/dense_res_block_2/output/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_2/output/kernel/Regularizer/add\
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:?????????:::::::::::P L
(
_output_shapes
:?????????
 
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
: 


__inference_loss_fn_0_16857301P
Lnetwork_arch_inner_encoder_kernel_regularizer_square_readvariableop_resource
identity
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpLnetwork_arch_inner_encoder_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpν
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
1network_arch/inner_encoder/kernel/Regularizer/Sum―
3network_arch/inner_encoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+225
3network_arch/inner_encoder/kernel/Regularizer/mul/x
1network_arch/inner_encoder/kernel/Regularizer/mulMul<network_arch/inner_encoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_encoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/mul―
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
Θ
v
0__inference_inner_decoder_layer_call_fn_16856819

inputs
unknown
identity’StatefulPartitionedCallΝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*(
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_inner_decoder_layer_call_and_return_conditional_losses_168568132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
ε
γ
&__inference_signature_wrapper_16862664
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

unknown_21
identity

identity_1

identity_2’StatefulPartitionedCall
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
unknown_21*#
Tin
2*
Tout
2*\
_output_shapesJ
H:?????????3:?????????3:?????????2*9
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*,
f'R%
#__inference__wrapped_model_168625112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????32

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????32

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesv
t:?????????3:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????3
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
: 


4__inference_dense_res_block_2_layer_call_fn_16856798

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
	unknown_8
identity’StatefulPartitionedCallΖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*(
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_dense_res_block_2_layer_call_and_return_conditional_losses_168567832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
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
: 
ΰ
h
__inference_loss_fn_9_168628505
1kernel_regularizer_square_readvariableop_resource
identityΘ
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
 *wΜ+22
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

μ
/__inference_network_arch_layer_call_fn_16860959
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

unknown_21
identity

identity_1

identity_2’StatefulPartitionedCall­
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
unknown_21*#
Tin
2*
Tout
2*\
_output_shapesJ
H:?????????3:?????????3:?????????2*9
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_network_arch_layer_call_and_return_conditional_losses_168608312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????32

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????32

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesv
t:?????????3:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????3
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
: 
ΰ
h
__inference_loss_fn_7_168628245
1kernel_regularizer_square_readvariableop_resource
identityΘ
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
 *wΜ+22
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
y
©
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_16856534

inputs*
&hidden0_matmul_readvariableop_resource+
'hidden0_biasadd_readvariableop_resource*
&hidden1_matmul_readvariableop_resource+
'hidden1_biasadd_readvariableop_resource*
&hidden2_matmul_readvariableop_resource+
'hidden2_biasadd_readvariableop_resource*
&hidden3_matmul_readvariableop_resource+
'hidden3_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity§
hidden0/MatMul/ReadVariableOpReadVariableOp&hidden0_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden0/MatMul/ReadVariableOp
hidden0/MatMulMatMulinputs%hidden0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden0/MatMul₯
hidden0/BiasAdd/ReadVariableOpReadVariableOp'hidden0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden0/BiasAdd/ReadVariableOp’
hidden0/BiasAddBiasAddhidden0/MatMul:product:0&hidden0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden0/BiasAddq
hidden0/ReluReluhidden0/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden0/Relu§
hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden1/MatMul/ReadVariableOp 
hidden1/MatMulMatMulhidden0/Relu:activations:0%hidden1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden1/MatMul₯
hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden1/BiasAdd/ReadVariableOp’
hidden1/BiasAddBiasAddhidden1/MatMul:product:0&hidden1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden1/BiasAddq
hidden1/ReluReluhidden1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden1/Relu§
hidden2/MatMul/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden2/MatMul/ReadVariableOp 
hidden2/MatMulMatMulhidden1/Relu:activations:0%hidden2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden2/MatMul₯
hidden2/BiasAdd/ReadVariableOpReadVariableOp'hidden2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden2/BiasAdd/ReadVariableOp’
hidden2/BiasAddBiasAddhidden2/MatMul:product:0&hidden2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden2/BiasAddq
hidden2/ReluReluhidden2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden2/Relu§
hidden3/MatMul/ReadVariableOpReadVariableOp&hidden3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden3/MatMul/ReadVariableOp 
hidden3/MatMulMatMulhidden2/Relu:activations:0%hidden3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden3/MatMul₯
hidden3/BiasAdd/ReadVariableOpReadVariableOp'hidden3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden3/BiasAdd/ReadVariableOp’
hidden3/BiasAddBiasAddhidden3/MatMul:product:0&hidden3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden3/BiasAddq
hidden3/ReluReluhidden3/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden3/Relu€
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMulhidden3/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
output/MatMul’
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
output/BiasAddg
addAddV2inputsoutput/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
add
Onetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden0_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_3/hidden0/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_3/hidden0/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/add
Onetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_3/hidden1/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_3/hidden1/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/add
Onetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_3/hidden2/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_3/hidden2/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/add
Onetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_3/hidden3/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_3/hidden3/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/add
Nnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02P
Nnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Square/ReadVariableOp
?network_arch/dense_res_block_3/output/kernel/Regularizer/SquareSquareVnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2A
?network_arch/dense_res_block_3/output/kernel/Regularizer/SquareΡ
>network_arch/dense_res_block_3/output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2@
>network_arch/dense_res_block_3/output/kernel/Regularizer/Const²
<network_arch/dense_res_block_3/output/kernel/Regularizer/SumSumCnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Square:y:0Gnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_3/output/kernel/Regularizer/SumΕ
>network_arch/dense_res_block_3/output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22@
>network_arch/dense_res_block_3/output/kernel/Regularizer/mul/x΄
<network_arch/dense_res_block_3/output/kernel/Regularizer/mulMulGnetwork_arch/dense_res_block_3/output/kernel/Regularizer/mul/x:output:0Enetwork_arch/dense_res_block_3/output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_3/output/kernel/Regularizer/mulΕ
>network_arch/dense_res_block_3/output/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>network_arch/dense_res_block_3/output/kernel/Regularizer/add/x±
<network_arch/dense_res_block_3/output/kernel/Regularizer/addAddV2Gnetwork_arch/dense_res_block_3/output/kernel/Regularizer/add/x:output:0@network_arch/dense_res_block_3/output/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_3/output/kernel/Regularizer/add\
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:?????????:::::::::::P L
(
_output_shapes
:?????????
 
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
: 
α
i
__inference_loss_fn_10_168628635
1kernel_regularizer_square_readvariableop_resource
identityΘ
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
 *wΜ+22
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
Τ

K__inference_inner_encoder_layer_call_and_return_conditional_losses_16857451

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
:?????????2
MatMulκ
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_encoder/kernel/Regularizer/Square/ReadVariableOpν
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
1network_arch/inner_encoder/kernel/Regularizer/Sum―
3network_arch/inner_encoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+225
3network_arch/inner_encoder/kernel/Regularizer/mul/x
1network_arch/inner_encoder/kernel/Regularizer/mulMul<network_arch/inner_encoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_encoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_encoder/kernel/Regularizer/mul―
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
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
y
©
O__inference_dense_res_block_2_layer_call_and_return_conditional_losses_16857166

inputs*
&hidden0_matmul_readvariableop_resource+
'hidden0_biasadd_readvariableop_resource*
&hidden1_matmul_readvariableop_resource+
'hidden1_biasadd_readvariableop_resource*
&hidden2_matmul_readvariableop_resource+
'hidden2_biasadd_readvariableop_resource*
&hidden3_matmul_readvariableop_resource+
'hidden3_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity§
hidden0/MatMul/ReadVariableOpReadVariableOp&hidden0_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden0/MatMul/ReadVariableOp
hidden0/MatMulMatMulinputs%hidden0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden0/MatMul₯
hidden0/BiasAdd/ReadVariableOpReadVariableOp'hidden0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden0/BiasAdd/ReadVariableOp’
hidden0/BiasAddBiasAddhidden0/MatMul:product:0&hidden0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden0/BiasAddq
hidden0/ReluReluhidden0/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden0/Relu§
hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden1/MatMul/ReadVariableOp 
hidden1/MatMulMatMulhidden0/Relu:activations:0%hidden1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden1/MatMul₯
hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden1/BiasAdd/ReadVariableOp’
hidden1/BiasAddBiasAddhidden1/MatMul:product:0&hidden1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden1/BiasAddq
hidden1/ReluReluhidden1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden1/Relu§
hidden2/MatMul/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden2/MatMul/ReadVariableOp 
hidden2/MatMulMatMulhidden1/Relu:activations:0%hidden2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden2/MatMul₯
hidden2/BiasAdd/ReadVariableOpReadVariableOp'hidden2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden2/BiasAdd/ReadVariableOp’
hidden2/BiasAddBiasAddhidden2/MatMul:product:0&hidden2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden2/BiasAddq
hidden2/ReluReluhidden2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden2/Relu§
hidden3/MatMul/ReadVariableOpReadVariableOp&hidden3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden3/MatMul/ReadVariableOp 
hidden3/MatMulMatMulhidden2/Relu:activations:0%hidden3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden3/MatMul₯
hidden3/BiasAdd/ReadVariableOpReadVariableOp'hidden3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden3/BiasAdd/ReadVariableOp’
hidden3/BiasAddBiasAddhidden3/MatMul:product:0&hidden3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden3/BiasAddq
hidden3/ReluReluhidden3/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden3/Relu€
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMulhidden3/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
output/MatMul’
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
output/BiasAddg
addAddV2inputsoutput/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
add
Onetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden0_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_2/hidden0/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_2/hidden0/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_2/hidden0/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_2/hidden0/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden0/kernel/Regularizer/add
Onetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_2/hidden1/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_2/hidden1/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_2/hidden1/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_2/hidden1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden1/kernel/Regularizer/add
Onetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_2/hidden2/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_2/hidden2/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_2/hidden2/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_2/hidden2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden2/kernel/Regularizer/add
Onetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_2/hidden3/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_2/hidden3/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_2/hidden3/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_2/hidden3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_2/hidden3/kernel/Regularizer/add
Nnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02P
Nnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Square/ReadVariableOp
?network_arch/dense_res_block_2/output/kernel/Regularizer/SquareSquareVnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2A
?network_arch/dense_res_block_2/output/kernel/Regularizer/SquareΡ
>network_arch/dense_res_block_2/output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2@
>network_arch/dense_res_block_2/output/kernel/Regularizer/Const²
<network_arch/dense_res_block_2/output/kernel/Regularizer/SumSumCnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Square:y:0Gnetwork_arch/dense_res_block_2/output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_2/output/kernel/Regularizer/SumΕ
>network_arch/dense_res_block_2/output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22@
>network_arch/dense_res_block_2/output/kernel/Regularizer/mul/x΄
<network_arch/dense_res_block_2/output/kernel/Regularizer/mulMulGnetwork_arch/dense_res_block_2/output/kernel/Regularizer/mul/x:output:0Enetwork_arch/dense_res_block_2/output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_2/output/kernel/Regularizer/mulΕ
>network_arch/dense_res_block_2/output/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>network_arch/dense_res_block_2/output/kernel/Regularizer/add/x±
<network_arch/dense_res_block_2/output/kernel/Regularizer/addAddV2Gnetwork_arch/dense_res_block_2/output/kernel/Regularizer/add/x:output:0@network_arch/dense_res_block_2/output/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_2/output/kernel/Regularizer/add\
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:?????????:::::::::::P L
(
_output_shapes
:?????????
 
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
: 
~
©
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_16856633

inputs*
&hidden0_matmul_readvariableop_resource+
'hidden0_biasadd_readvariableop_resource*
&hidden1_matmul_readvariableop_resource+
'hidden1_biasadd_readvariableop_resource*
&hidden2_matmul_readvariableop_resource+
'hidden2_biasadd_readvariableop_resource*
&hidden3_matmul_readvariableop_resource+
'hidden3_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity§
hidden0/MatMul/ReadVariableOpReadVariableOp&hidden0_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden0/MatMul/ReadVariableOp
hidden0/MatMulMatMulinputs%hidden0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden0/MatMul₯
hidden0/BiasAdd/ReadVariableOpReadVariableOp'hidden0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden0/BiasAdd/ReadVariableOp’
hidden0/BiasAddBiasAddhidden0/MatMul:product:0&hidden0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden0/BiasAddq
hidden0/ReluReluhidden0/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden0/Relu
hidden0/IdentityIdentityhidden0/Relu:activations:0*
T0*(
_output_shapes
:?????????2
hidden0/Identity§
hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden1/MatMul/ReadVariableOp
hidden1/MatMulMatMulhidden0/Identity:output:0%hidden1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden1/MatMul₯
hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden1/BiasAdd/ReadVariableOp’
hidden1/BiasAddBiasAddhidden1/MatMul:product:0&hidden1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden1/BiasAddq
hidden1/ReluReluhidden1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden1/Relu
hidden1/IdentityIdentityhidden1/Relu:activations:0*
T0*(
_output_shapes
:?????????2
hidden1/Identity§
hidden2/MatMul/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden2/MatMul/ReadVariableOp
hidden2/MatMulMatMulhidden1/Identity:output:0%hidden2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden2/MatMul₯
hidden2/BiasAdd/ReadVariableOpReadVariableOp'hidden2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden2/BiasAdd/ReadVariableOp’
hidden2/BiasAddBiasAddhidden2/MatMul:product:0&hidden2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden2/BiasAddq
hidden2/ReluReluhidden2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden2/Relu
hidden2/IdentityIdentityhidden2/Relu:activations:0*
T0*(
_output_shapes
:?????????2
hidden2/Identity§
hidden3/MatMul/ReadVariableOpReadVariableOp&hidden3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
hidden3/MatMul/ReadVariableOp
hidden3/MatMulMatMulhidden2/Identity:output:0%hidden3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden3/MatMul₯
hidden3/BiasAdd/ReadVariableOpReadVariableOp'hidden3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
hidden3/BiasAdd/ReadVariableOp’
hidden3/BiasAddBiasAddhidden3/MatMul:product:0&hidden3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden3/BiasAddq
hidden3/ReluReluhidden3/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden3/Relu
hidden3/IdentityIdentityhidden3/Relu:activations:0*
T0*(
_output_shapes
:?????????2
hidden3/Identity€
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMulhidden3/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
output/MatMul’
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
output/BiasAddz
output/IdentityIdentityoutput/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
output/Identityh
addAddV2inputsoutput/Identity:output:0*
T0*(
_output_shapes
:?????????2
add
Onetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden0_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_3/hidden0/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_3/hidden0/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_3/hidden0/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_3/hidden0/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden0/kernel/Regularizer/add
Onetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_3/hidden1/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_3/hidden1/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_3/hidden1/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_3/hidden1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden1/kernel/Regularizer/add
Onetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_3/hidden2/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_3/hidden2/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_3/hidden2/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_3/hidden2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden2/kernel/Regularizer/add
Onetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&hidden3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Q
Onetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Square/ReadVariableOp
@network_arch/dense_res_block_3/hidden3/kernel/Regularizer/SquareSquareWnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2B
@network_arch/dense_res_block_3/hidden3/kernel/Regularizer/SquareΣ
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2A
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/ConstΆ
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/SumSumDnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Square:y:0Hnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/SumΗ
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22A
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/mul/xΈ
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/mulMulHnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/mul/x:output:0Fnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/mulΗ
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?network_arch/dense_res_block_3/hidden3/kernel/Regularizer/add/x΅
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/addAddV2Hnetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/add/x:output:0Anetwork_arch/dense_res_block_3/hidden3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2?
=network_arch/dense_res_block_3/hidden3/kernel/Regularizer/add
Nnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02P
Nnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Square/ReadVariableOp
?network_arch/dense_res_block_3/output/kernel/Regularizer/SquareSquareVnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2A
?network_arch/dense_res_block_3/output/kernel/Regularizer/SquareΡ
>network_arch/dense_res_block_3/output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2@
>network_arch/dense_res_block_3/output/kernel/Regularizer/Const²
<network_arch/dense_res_block_3/output/kernel/Regularizer/SumSumCnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Square:y:0Gnetwork_arch/dense_res_block_3/output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_3/output/kernel/Regularizer/SumΕ
>network_arch/dense_res_block_3/output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+22@
>network_arch/dense_res_block_3/output/kernel/Regularizer/mul/x΄
<network_arch/dense_res_block_3/output/kernel/Regularizer/mulMulGnetwork_arch/dense_res_block_3/output/kernel/Regularizer/mul/x:output:0Enetwork_arch/dense_res_block_3/output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_3/output/kernel/Regularizer/mulΕ
>network_arch/dense_res_block_3/output/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>network_arch/dense_res_block_3/output/kernel/Regularizer/add/x±
<network_arch/dense_res_block_3/output/kernel/Regularizer/addAddV2Gnetwork_arch/dense_res_block_3/output/kernel/Regularizer/add/x:output:0@network_arch/dense_res_block_3/output/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2>
<network_arch/dense_res_block_3/output/kernel/Regularizer/add\
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:?????????:::::::::::P L
(
_output_shapes
:?????????
 
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
: 
α
i
__inference_loss_fn_11_168628765
1kernel_regularizer_square_readvariableop_resource
identityΘ
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
 *wΜ+22
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


__inference_loss_fn_1_16857352P
Lnetwork_arch_inner_decoder_kernel_regularizer_square_readvariableop_resource
identity
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpReadVariableOpLnetwork_arch_inner_decoder_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype02E
Cnetwork_arch/inner_decoder/kernel/Regularizer/Square/ReadVariableOpν
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
1network_arch/inner_decoder/kernel/Regularizer/Sum―
3network_arch/inner_decoder/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *wΜ+225
3network_arch/inner_decoder/kernel/Regularizer/mul/x
1network_arch/inner_decoder/kernel/Regularizer/mulMul<network_arch/inner_decoder/kernel/Regularizer/mul/x:output:0:network_arch/inner_decoder/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 23
1network_arch/inner_decoder/kernel/Regularizer/mul―
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

θ
+__inference_restored_function_body_16862458
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

unknown_21
identity

identity_1

identity_2’StatefulPartitionedCall­
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
unknown_21*#
Tin
2*
Tout
2*\
_output_shapesJ
H:?????????3:?????????3:?????????2*9
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_network_arch_layer_call_and_return_conditional_losses_168608312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????32

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????32

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesv
t:?????????3:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????3
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
: 
ΰ
h
__inference_loss_fn_2_168627595
1kernel_regularizer_square_readvariableop_resource
identityΘ
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
 *wΜ+22
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
: "―L
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
serving_default_input_1:0?????????3A
output_15
StatefulPartitionedCall:0?????????3A
output_25
StatefulPartitionedCall:1?????????3A
output_35
StatefulPartitionedCall:2?????????2tensorflow/serving/predict:Κ«
α	
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

	variables
regularization_losses
trainable_variables
	keras_api
+³&call_and_return_all_conditional_losses
΄_default_save_signature
΅__call__"
_tf_keras_model{"class_name": "NetworkArch", "name": "network_arch", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "NetworkArch"}, "training_config": {"loss": [{"class_name": "RelMSE", "config": {"reduction": "auto", "name": null}}, {"class_name": "RelMSE", "config": {"reduction": "auto", "name": null}}, {"class_name": "RelMSE", "config": {"reduction": "auto", "name": null}}], "metrics": null, "weighted_metrics": null, "loss_weights": [1, 1, 1], "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0008818680071271956, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}


layers
	variables
regularization_losses
trainable_variables
	keras_api
+Ά&call_and_return_all_conditional_losses
·__call__"ώ
_tf_keras_layerδ{"class_name": "DenseResBlock", "name": "dense_res_block_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"layer was saved without config": true}}


layers
	variables
regularization_losses
trainable_variables
	keras_api
+Έ&call_and_return_all_conditional_losses
Ή__call__"ώ
_tf_keras_layerδ{"class_name": "DenseResBlock", "name": "dense_res_block_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"layer was saved without config": true}}
§

kernel
	variables
regularization_losses
trainable_variables
	keras_api
+Ί&call_and_return_all_conditional_losses
»__call__"
_tf_keras_layerπ{"class_name": "Dense", "name": "inner_encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "inner_encoder", "trainable": true, "dtype": "float32", "units": 21, "activation": "linear", "use_bias": false, "kernel_initializer": "identity_init", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
:2Variable
§

kernel
	variables
regularization_losses
 trainable_variables
!	keras_api
+Ό&call_and_return_all_conditional_losses
½__call__"
_tf_keras_layerπ{"class_name": "Dense", "name": "inner_decoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "inner_decoder", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": false, "kernel_initializer": "identity_init", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 21}}}}
 "
trackable_list_wrapper
"
	optimizer
 "
trackable_list_wrapper
-
Ύserving_default"
signature_map
Ξ
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
20
21
22"
trackable_list_wrapper
0
Ώ0
ΐ1"
trackable_list_wrapper
Ξ
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
20
21
22"
trackable_list_wrapper
Ξ
6metrics

	variables
regularization_losses
7layer_regularization_losses
8non_trainable_variables

9layers
:layer_metrics
trainable_variables
΅__call__
΄_default_save_signature
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
C
;0
<1
=2
>3
?4"
trackable_list_wrapper
f
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9"
trackable_list_wrapper
H
Α0
Β1
Γ2
Δ3
Ε4"
trackable_list_wrapper
f
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9"
trackable_list_wrapper
°
@metrics
	variables
regularization_losses
Alayer_regularization_losses
Bnon_trainable_variables

Clayers
Dlayer_metrics
trainable_variables
·__call__
+Ά&call_and_return_all_conditional_losses
'Ά"call_and_return_conditional_losses"
_generic_user_object
C
E0
F1
G2
H3
I4"
trackable_list_wrapper
f
,0
-1
.2
/3
04
15
26
37
48
59"
trackable_list_wrapper
H
Ζ0
Η1
Θ2
Ι3
Κ4"
trackable_list_wrapper
f
,0
-1
.2
/3
04
15
26
37
48
59"
trackable_list_wrapper
°
Jmetrics
	variables
regularization_losses
Klayer_regularization_losses
Lnon_trainable_variables

Mlayers
Nlayer_metrics
trainable_variables
Ή__call__
+Έ&call_and_return_all_conditional_losses
'Έ"call_and_return_conditional_losses"
_generic_user_object
4:2	2!network_arch/inner_encoder/kernel
'
0"
trackable_list_wrapper
(
Ώ0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
Ometrics
	variables
regularization_losses
Player_regularization_losses
Qnon_trainable_variables

Rlayers
Slayer_metrics
trainable_variables
»__call__
+Ί&call_and_return_all_conditional_losses
'Ί"call_and_return_conditional_losses"
_generic_user_object
4:2	2!network_arch/inner_decoder/kernel
'
0"
trackable_list_wrapper
(
ΐ0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
Tmetrics
	variables
regularization_losses
Ulayer_regularization_losses
Vnon_trainable_variables

Wlayers
Xlayer_metrics
 trainable_variables
½__call__
+Ό&call_and_return_all_conditional_losses
'Ό"call_and_return_conditional_losses"
_generic_user_object
A:?
2-network_arch/dense_res_block_2/hidden0/kernel
::82+network_arch/dense_res_block_2/hidden0/bias
A:?
2-network_arch/dense_res_block_2/hidden1/kernel
::82+network_arch/dense_res_block_2/hidden1/bias
A:?
2-network_arch/dense_res_block_2/hidden2/kernel
::82+network_arch/dense_res_block_2/hidden2/bias
A:?
2-network_arch/dense_res_block_2/hidden3/kernel
::82+network_arch/dense_res_block_2/hidden3/bias
@:>
2,network_arch/dense_res_block_2/output/kernel
9:72*network_arch/dense_res_block_2/output/bias
A:?
2-network_arch/dense_res_block_3/hidden0/kernel
::82+network_arch/dense_res_block_3/hidden0/bias
A:?
2-network_arch/dense_res_block_3/hidden1/kernel
::82+network_arch/dense_res_block_3/hidden1/bias
A:?
2-network_arch/dense_res_block_3/hidden2/kernel
::82+network_arch/dense_res_block_3/hidden2/bias
A:?
2-network_arch/dense_res_block_3/hidden3/kernel
::82+network_arch/dense_res_block_3/hidden3/bias
@:>
2,network_arch/dense_res_block_3/output/kernel
9:72*network_arch/dense_res_block_3/output/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
έ

"kernel
#bias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
+Λ&call_and_return_all_conditional_losses
Μ__call__"Ά
_tf_keras_layer{"class_name": "Dense", "name": "hidden0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "hidden0", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128]}}
έ

$kernel
%bias
]	variables
^regularization_losses
_trainable_variables
`	keras_api
+Ν&call_and_return_all_conditional_losses
Ξ__call__"Ά
_tf_keras_layer{"class_name": "Dense", "name": "hidden1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128]}}
έ

&kernel
'bias
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
+Ο&call_and_return_all_conditional_losses
Π__call__"Ά
_tf_keras_layer{"class_name": "Dense", "name": "hidden2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128]}}
έ

(kernel
)bias
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
+Ρ&call_and_return_all_conditional_losses
?__call__"Ά
_tf_keras_layer{"class_name": "Dense", "name": "hidden3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "hidden3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128]}}
έ

*kernel
+bias
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
+Σ&call_and_return_all_conditional_losses
Τ__call__"Ά
_tf_keras_layer{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
;0
<1
=2
>3
?4"
trackable_list_wrapper
 "
trackable_dict_wrapper
έ

,kernel
-bias
m	variables
nregularization_losses
otrainable_variables
p	keras_api
+Υ&call_and_return_all_conditional_losses
Φ__call__"Ά
_tf_keras_layer{"class_name": "Dense", "name": "hidden0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "hidden0", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128]}}
έ

.kernel
/bias
q	variables
rregularization_losses
strainable_variables
t	keras_api
+Χ&call_and_return_all_conditional_losses
Ψ__call__"Ά
_tf_keras_layer{"class_name": "Dense", "name": "hidden1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128]}}
έ

0kernel
1bias
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
+Ω&call_and_return_all_conditional_losses
Ϊ__call__"Ά
_tf_keras_layer{"class_name": "Dense", "name": "hidden2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128]}}
έ

2kernel
3bias
y	variables
zregularization_losses
{trainable_variables
|	keras_api
+Ϋ&call_and_return_all_conditional_losses
ά__call__"Ά
_tf_keras_layer{"class_name": "Dense", "name": "hidden3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "hidden3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128]}}
ή

4kernel
5bias
}	variables
~regularization_losses
trainable_variables
	keras_api
+έ&call_and_return_all_conditional_losses
ή__call__"Ά
_tf_keras_layer{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.99999993922529e-09}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1632, 128]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
E0
F1
G2
H3
I4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
Ώ0"
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
ΐ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
"0
#1"
trackable_list_wrapper
(
Α0"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
΅
metrics
Y	variables
Zregularization_losses
 layer_regularization_losses
non_trainable_variables
layers
layer_metrics
[trainable_variables
Μ__call__
+Λ&call_and_return_all_conditional_losses
'Λ"call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
(
Β0"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
΅
metrics
]	variables
^regularization_losses
 layer_regularization_losses
non_trainable_variables
layers
layer_metrics
_trainable_variables
Ξ__call__
+Ν&call_and_return_all_conditional_losses
'Ν"call_and_return_conditional_losses"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
(
Γ0"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
΅
metrics
a	variables
bregularization_losses
 layer_regularization_losses
non_trainable_variables
layers
layer_metrics
ctrainable_variables
Π__call__
+Ο&call_and_return_all_conditional_losses
'Ο"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
(
Δ0"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
΅
metrics
e	variables
fregularization_losses
 layer_regularization_losses
non_trainable_variables
layers
layer_metrics
gtrainable_variables
?__call__
+Ρ&call_and_return_all_conditional_losses
'Ρ"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
(
Ε0"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
΅
metrics
i	variables
jregularization_losses
 layer_regularization_losses
non_trainable_variables
layers
layer_metrics
ktrainable_variables
Τ__call__
+Σ&call_and_return_all_conditional_losses
'Σ"call_and_return_conditional_losses"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
(
Ζ0"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
΅
metrics
m	variables
nregularization_losses
 layer_regularization_losses
non_trainable_variables
layers
layer_metrics
otrainable_variables
Φ__call__
+Υ&call_and_return_all_conditional_losses
'Υ"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
(
Η0"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
΅
metrics
q	variables
rregularization_losses
  layer_regularization_losses
‘non_trainable_variables
’layers
£layer_metrics
strainable_variables
Ψ__call__
+Χ&call_and_return_all_conditional_losses
'Χ"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
(
Θ0"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
΅
€metrics
u	variables
vregularization_losses
 ₯layer_regularization_losses
¦non_trainable_variables
§layers
¨layer_metrics
wtrainable_variables
Ϊ__call__
+Ω&call_and_return_all_conditional_losses
'Ω"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
(
Ι0"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
΅
©metrics
y	variables
zregularization_losses
 ͺlayer_regularization_losses
«non_trainable_variables
¬layers
­layer_metrics
{trainable_variables
ά__call__
+Ϋ&call_and_return_all_conditional_losses
'Ϋ"call_and_return_conditional_losses"
_generic_user_object
.
40
51"
trackable_list_wrapper
(
Κ0"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
΅
?metrics
}	variables
~regularization_losses
 ―layer_regularization_losses
°non_trainable_variables
±layers
²layer_metrics
trainable_variables
ή__call__
+έ&call_and_return_all_conditional_losses
'έ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
Α0"
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
Β0"
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
Γ0"
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
Δ0"
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
Ε0"
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
Ζ0"
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
Η0"
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
Θ0"
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
Ι0"
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
Κ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
2
J__inference_network_arch_layer_call_and_return_conditional_losses_16860831Α
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
annotationsͺ *+’(
&#
input_1?????????3
ζ2γ
#__inference__wrapped_model_16862511»
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
annotationsͺ *+’(
&#
input_1?????????3
ψ2υ
/__inference_network_arch_layer_call_fn_16860959Α
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
annotationsͺ *+’(
&#
input_1?????????3
ο2μ
O__inference_dense_res_block_2_layer_call_and_return_conditional_losses_16857166
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
annotationsͺ *
 
Τ2Ρ
4__inference_dense_res_block_2_layer_call_fn_16856798
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
annotationsͺ *
 
ο2μ
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_16856534
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
annotationsͺ *
 
Τ2Ρ
4__inference_dense_res_block_3_layer_call_fn_16856648
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
annotationsͺ *
 
λ2θ
K__inference_inner_encoder_layer_call_and_return_conditional_losses_16856549
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
annotationsͺ *
 
Π2Ν
0__inference_inner_encoder_layer_call_fn_16860965
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
annotationsͺ *
 
λ2θ
K__inference_inner_decoder_layer_call_and_return_conditional_losses_16857265
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
annotationsͺ *
 
Π2Ν
0__inference_inner_decoder_layer_call_fn_16856819
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
annotationsͺ *
 
5B3
&__inference_signature_wrapper_16862664input_1
΅2²
__inference_loss_fn_0_16857301
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
annotationsͺ *’ 
΅2²
__inference_loss_fn_1_16857352
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
annotationsͺ *’ 
΅2²
__inference_loss_fn_2_16862759
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
annotationsͺ *’ 
΅2²
__inference_loss_fn_3_16862772
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
annotationsͺ *’ 
΅2²
__inference_loss_fn_4_16862785
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
annotationsͺ *’ 
΅2²
__inference_loss_fn_5_16862798
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
annotationsͺ *’ 
΅2²
__inference_loss_fn_6_16862811
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
annotationsͺ *’ 
΅2²
__inference_loss_fn_7_16862824
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
annotationsͺ *’ 
΅2²
__inference_loss_fn_8_16862837
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
annotationsͺ *’ 
΅2²
__inference_loss_fn_9_16862850
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
annotationsͺ *’ 
Ά2³
__inference_loss_fn_10_16862863
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
annotationsͺ *’ 
Ά2³
__inference_loss_fn_11_16862876
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
annotationsͺ *’ 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
#__inference__wrapped_model_16862511φ"#$%&'()*+,-./0123455’2
+’(
&#
input_1?????????3
ͺ "£ͺ
3
output_1'$
output_1?????????3
3
output_2'$
output_2?????????3
3
output_3'$
output_3?????????2Ή
O__inference_dense_res_block_2_layer_call_and_return_conditional_losses_16857166f
"#$%&'()*+0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
4__inference_dense_res_block_2_layer_call_fn_16856798Y
"#$%&'()*+0’-
&’#
!
inputs?????????
ͺ "?????????Ή
O__inference_dense_res_block_3_layer_call_and_return_conditional_losses_16856534f
,-./0123450’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
4__inference_dense_res_block_3_layer_call_fn_16856648Y
,-./0123450’-
&’#
!
inputs?????????
ͺ "?????????«
K__inference_inner_decoder_layer_call_and_return_conditional_losses_16857265\/’,
%’"
 
inputs?????????
ͺ "&’#

0?????????
 
0__inference_inner_decoder_layer_call_fn_16856819O/’,
%’"
 
inputs?????????
ͺ "?????????«
K__inference_inner_encoder_layer_call_and_return_conditional_losses_16856549\0’-
&’#
!
inputs?????????
ͺ "%’"

0?????????
 
0__inference_inner_encoder_layer_call_fn_16860965O0’-
&’#
!
inputs?????????
ͺ "?????????=
__inference_loss_fn_0_16857301’

’ 
ͺ " >
__inference_loss_fn_10_168628632’

’ 
ͺ " >
__inference_loss_fn_11_168628764’

’ 
ͺ " =
__inference_loss_fn_1_16857352’

’ 
ͺ " =
__inference_loss_fn_2_16862759"’

’ 
ͺ " =
__inference_loss_fn_3_16862772$’

’ 
ͺ " =
__inference_loss_fn_4_16862785&’

’ 
ͺ " =
__inference_loss_fn_5_16862798(’

’ 
ͺ " =
__inference_loss_fn_6_16862811*’

’ 
ͺ " =
__inference_loss_fn_7_16862824,’

’ 
ͺ " =
__inference_loss_fn_8_16862837.’

’ 
ͺ " =
__inference_loss_fn_9_168628500’

’ 
ͺ " 
J__inference_network_arch_layer_call_and_return_conditional_losses_16860831Λ"#$%&'()*+,-./0123455’2
+’(
&#
input_1?????????3
ͺ "y’v
o’l
"
0/0?????????3
"
0/1?????????3
"
0/2?????????2
 ο
/__inference_network_arch_layer_call_fn_16860959»"#$%&'()*+,-./0123455’2
+’(
&#
input_1?????????3
ͺ "i’f
 
0?????????3
 
1?????????3
 
2?????????2¬
&__inference_signature_wrapper_16862664"#$%&'()*+,-./012345@’=
’ 
6ͺ3
1
input_1&#
input_1?????????3"£ͺ
3
output_1'$
output_1?????????3
3
output_2'$
output_2?????????3
3
output_3'$
output_3?????????2