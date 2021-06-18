### pytorch 教程

#### pytorch 保存和加载模型权重方式

```shell
`保存和加载整个模型`
# 保存模型
torch.save(model, 'model.pkl')

# 模型加载
model = torch.load('model.pkl')

`tips：`这种方式无需自定义网络，保存时已把网络结构保存
```



```shell 
`仅仅保存模型参数以及分别加载模型结构和参数`
# 模型参数保存
torch.save(model.state_dict(), 'model_param.pkl')

# 模型参数加载(先加载模型结构,可以是自定义的，也可以是已知，然后再加载模型参数)
model = torchvision.models.resnet50() 
model.load_state_dict(torch.load('model_param.pkl'))

`tips:` 这种方式需要自己定义网络，并且其中的参数名称和结构要与保存的模型中的一致（可以是部分网络），相对灵活，便于对网络进行修改。
```



```shell
`CPU模型加载GPU参数`
model.load_state_dict(torch.load(’model_param.pkl‘, map_loaction='cpu'))

`通过DataParalle使用多GPU`
model = DataParalle(model)

# 保存参数
torch.save(model.module.state_dict(), 'model_param.pkl')
```



#### pytorch 加载预训练模型

```shell
`加载预训练模型和参数`
resnet50 = models.resnet(pretrained=True)

`只加载模型，不加载预训练模型参数`
# 加载模型
resnet18 = models.resnet18(pretrained=False)

# 加载预先下载好的预训练模型参数
resnet18.load_state_dict(torch.load('resnet18-5c106cde.pth'))


`加载部分预训练模型`
resnet152 = models.resnet152(pretrained=True)
pretrained_dict = resnet152.state_dict()
​```
	加载torchvision中的预训练模型和参数后，通过state_dict()方法获取参数，
	也可以直接从官方model_zoo下载
​```
model_dict = model.state_dict()

#将pretrained_dict里不属于model_dict的键剔除掉
preTrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

# 更新现有的model_dict
model_dict.update(pretrained_dict)

# 加载我们真正需要的state_dict
model.load_state_dict(model_dict)




```



>







