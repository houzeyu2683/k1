
'不管發生什麼事情，相信你的交叉驗證，堅持這個信念。'

import data
import network

'載入資料，預設切成 train, exam, test 三個部份，其中 exam 是有 target 的測試集。'
table = data.table(path="./SOURCE/CSV/ANNOTATION.csv", target='classname', split=0.1, seed=0)
train, exam, test = table.read()

'使用 train 來進行訓練，交叉驗證的測試集為 check 。'
validation = data.validation(table=train, target='subject', size=10, seed=0)
validation.fold()

for i in range(validation.size):  # i = 0
    
    validation.get(index=i)
    validation.train = data.dataset(table=validation.train, process=data.process)
    validation.check = data.dataset(table=validation.check, process=data.process)
    loader = data.loader(train=validation.train, check=validation.check, batch=32)
    pass

    model     = network.model()
    criterion = network.criterion.cel()
    optimizer = network.optimizer.adam(model)
    pass

    folder  = "SOURCE/LOG/FOLD-" + str(i)
    machine = network.machine(model=model, optimizer=optimizer, criterion=criterion, device='cuda', folder=folder, checkpoint="0")
    pass

    history = {
        'train' : {"loss":[]},
        'check' : {"loss":[]}
    }
    epoch = 40
    for e in range(epoch):  # e = 0

        machine.learn(loader=loader.train)
        machine.save("checkpoint")
        machine.update('schedule')
        machine.update('checkpoint')
        pass

        evaluation = machine.evaluate(loader=loader.train)
        history['train']['loss'] += [evaluation['loss']]
        pass

        evaluation = machine.evaluate(loader=loader.check)
        history['check']['loss'] += [evaluation['loss']]
        pass

        report = network.report(train=history['train'], check=history['check'])
        report.summarize()
        report.save(folder=folder)
        pass

    pass

'載入訓練模型對 exam, test 進行預測，此時可以參考 exam 的評估分數，這跟 test 上表現理論上要差不多。'
loader = data.loader(exam=data.dataset(table=exam, process=data.process), 
                     test=data.dataset(table=test, process=data.process), 
                     batch=32)

likelihood = {
    "exam":0,
    "test":0
}
for n, i in enumerate(["SOURCE/LOG/FOLD-"+str(i)+'/39.checkpoint' for i in range(10)], 1):  # i = 'SOURCE/LOG/FOLD-0/39.checkpoint'

    model = network.model()
    machine = network.machine(model=model, optimizer=None, criterion=None, device='cuda', folder=None, checkpoint=None)
    machine.load(what='weight', path=i)
    likelihood['exam'] += machine.predict(loader=loader.exam)
    likelihood['test'] += machine.predict(loader=loader.test)
    pass

likelihood['exam'] = likelihood['exam'] / n
likelihood['test'] = likelihood['test'] / n
pass

print("loss of exam: {}".format(network.metric.cel(target=exam['target'], likelihood=likelihood['exam'])))
pass

submission = test[['img']].join(likelihood['test'])
submission.to_csv("SOURCE/LOG/SUBMISSION.csv", index=False)
# submission.columns = ['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']