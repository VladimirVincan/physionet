import torch
import torch.nn as nn
import torch.optim as optim


def calculate_loss(feats, outputSignal, model, arousal_criterion, apnea_criterion, wake_criterion, settings):
        outputSignal = outputSignal[:, ::settings['reduction_factor'], :]
        arousalTargs = outputSignal[:, :, 0]
        apneaTargs = outputSignal[:, :, 1]
        wakeTargs = outputSignal[:, :, 2]

        feats = feats.permute(0, 2, 1)
        feats = feats.to(settings['device'])
        arousalTargs = arousalTargs.to(settings['device'])
        apneaTargs = apneaTargs.to(settings['device'])
        wakeTargs = wakeTargs.to(settings['device'])

        arousalOutputs, apneaHypopneaOutputs, sleepStageOutputs = model(feats) #Compute the network outputs on the batch, (x1, x2, x3) u model.py
        arousalOutputs = arousalOutputs.permute(0, 2, 1).contiguous().view(-1, 2).float() #Compute the losses
        arousalTargs = arousalTargs.permute(1, 0).contiguous().view(-1).long()
        apneaHypopneaOutputs = apneaHypopneaOutputs.permute(0, 2, 1).contiguous().view(-1, 2).float() #preuredjuje dimenzije za Loss funkciju
        apneaTargs = apneaTargs.permute(1, 0).contiguous().view(-1).long()
        sleepStageOutputs = sleepStageOutputs.permute(0, 2, 1).contiguous().view(-1, 2).float()
        wakeTargs = wakeTargs.permute(1, 0).contiguous().view(-1).long()

        arousalLoss = arousal_criterion(arousalOutputs, arousalTargs) #prosledjuje izlaze sa ocekivanim i racuna gubitak
        apneaHypopneaLoss = apnea_criterion(apneaHypopneaOutputs, apneaTargs)
        sleepStageLoss = wake_criterion(sleepStageOutputs, wakeTargs)

        loss = ((2*arousalLoss) + apneaHypopneaLoss + sleepStageLoss) / 4.0 #zbirni gubitak, where the target arousal loss weight is set to 2 and the weights of other task losses are set to 1, since the auxiliary tasks are less important than the desired task (pise u 0.54.pdf)

        return loss


def train_one_epoch(model, dataloader, criterion, scheduler, optimizer, settings):
    model.train()
    runningLoss = 0.0  # za zbir gubitka u epohi

    truths=[]
    fileNames=[]
    folders=[]
    # for idx in range(numBatchesPerEpoch):
    for batch_idx, (folderName, feats, outputSignal) in enumerate(dataloader):
        print('batch_idx: ' + str(batch_idx), flush=True)
        # truths.append(outputSignal[batch_idx])
        folders = list(folderName)

        optimizer.zero_grad() #resetovanje gradijenata
        loss = calculate_loss(feats, outputSignal, model, arousal_criterion, apnea_criterion, wake_criterion, settings)
        loss.backward() #Backpropagation izracunavanje gradijenata
        runningLoss += loss.data.cpu().numpy() #dodavanje trenutnog gubitka u ukupni

        optimizer.step() #azuriranje parametara modela

    # fileNames=[OUTPUT_PATH + '/' + folderName + '.vec' for folderName in folders]

    epoch_loss = runningLoss / float(settings['train_batch_size'])
    # epoch_auprc = count_aurpc(truths, fileNames)
    epoch_auprc = 0
    return epoch_loss, epoch_auprc


def validate(model, dataloader, criterion, settings):
    model.eval()
    runningLoss = 0.0

    truths=[]
    fileNames=[]

    with torch.no_grad():
        for _, (folderName, feats, outputSignal) in enumerate(dataloader):
            truths.append(outputSignal)
            fileNames.append(OUTPUT_PATH + '/' + folderName + '.vec')

            loss = calculate_loss(feats, outputSignal, model, arousalCriterion, apneaCriterion, wakeCriterion, settings)
            runningLoss += loss.data.cpu().numpy() #dodavanje trenutnog gubitka u ukupni

    epochLoss = runningLoss / float(settings['train_batch_size'])
    # epochAuprc = count_aurpc(truths, fileNames)
    epochAuprc = 0
    return epochLoss, epochAuprc


def train_loop(model, train_dataloader, validation_dataloader, settings):
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([settings['pos_weight']])).to(settings['device'])
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=settings['max_lr'],
                                              pct_start=settings['warmstart_percentage'],
                                              steps_per_epoch=int(
                                                  len(train_dataloader)),
                                              epochs=settings['epochs'],
                                              anneal_strategy='linear')

    for epoch in range(1, settings['epochs'] + 1):
        print('============ TRAIN EPOCH: ' + str(epoch) + ' ============', flush=True)
        trainLoss, trainAuprc = train_one_epoch(model, train_dataloader, criterion, scheduler, optimizer, settings)
        valLoss, valAuprc = validate(model, validation_dataloader, criterion, settings)

        checkpointPath = f'./TrainedModels/checkpoint_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpointPath)

