def training(train_dataloader, val_loader, network, loss_function, optimizer):
    import torch
    
    num_epochs = 5
    for epoch in range(num_epochs):
        network.train()
        train_loss = 0.0
        
        for (data, target) in train_dataloader:
            optimizer.zero_grad()
                                    
            # Forward pass
            predicted = network(data)
            
            # Calcolo loss
            loss = loss_function(predicted, target)
            
            # Backward pass e ottimizzazione
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calcola la loss media per epoch
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validazione
        network.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (data, target) in val_loader:                
                predicted = network(data)
                
                loss = loss_function(predicted, target)
                     
                val_loss += loss.item()
        
        # Calcola la loss media di validazione
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')

def x_training(train_dataloader, network):
    from PIL import Image
    for (data, _) in train_dataloader:                                
        predicted = network(data)
        
        return Image.open(predicted)
        
def test(network, loader, loss_function):
    import torch
    import torch.nn.functional as F

    network.eval()
    test_loss = 0.0
    correct_orientation = 0
    total = 0
    
    with torch.no_grad():
        for (data, target) in loader:
            predicted = network(data) 
            
            loss = loss_function(predicted, target)
            
            # Applica softmax per ottenere le probabilità
            probabilities = F.softmax(predicted, dim=1)
            
            # Ottieni l'indice della classe con la massima probabilità
            orientation_predicted = probabilities.argmax(dim=1)
            
            # Ottieni l'indice della classe corretta dal target
            orientation_target = target.argmax(dim=1)
            
            # Calcola le corrispondenze
            orientation_matches = (orientation_predicted == orientation_target).sum()
            correct_orientation += orientation_matches.item()
            
            print("Predicted:", orientation_predicted)
            print("Target:", orientation_target)
            
            total += data.size(0)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(loader)
    accuracy = 100 * correct_orientation / total
    
    print(f'Test Results:')
    print(f'Average Loss: {avg_test_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')
      
    return accuracy