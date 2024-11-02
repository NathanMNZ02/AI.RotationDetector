mean, std = 0.5, 0.5

def train():
    import torch
    import torch.utils
    import torchvision.transforms as transforms
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from Model.convolutional_network import ConvolutionalNetwork
    from Model.dataset import Easy4ProDataset
    from Model.procedure import training, x_training, test
        
    # Definisci le trasformazioni
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    # Carica i dataset
    train_dataset = Easy4ProDataset(
        working_dir='DataSet/train/',
        csv_path='DataSet/train/_classes.csv', 
        transform=transform
    )
    
    val_dataset = Easy4ProDataset(
        working_dir='DataSet/valid/',
        csv_path='DataSet/valid/_classes.csv', 
        transform=transform
    )
    
    test_dataset = Easy4ProDataset(
        working_dir='DataSet/test/',
        csv_path='DataSet/test/_classes.csv', 
        transform=transform
    )
    
    # Crea i data loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Inizializza il modello e l'ottimizzatore
    convolutional_network = ConvolutionalNetwork()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(convolutional_network.parameters(), lr=0.001)
    
    x_training(train_dataloader=train_loader,
        network=convolutional_network)
    
    ####
    # Training
    training(
        train_dataloader=train_loader,
        val_loader=val_loader,
        network=convolutional_network,
        loss_function=loss_function,
        optimizer=optimizer
    )
    
    # Test
    acc = test(
        network=convolutional_network,
        loader=test_loader,
        loss_function=loss_function
    )
    
    if acc >= 80:
        torch.save(convolutional_network.state_dict(), 'convolutional_weights.pth')
    ####

def predict(weights_path, image_path):
    import torch
    import torch.nn.functional as F
    
    from PIL import Image
    from torchvision import transforms
    from Model.convolutional_network import ConvolutionalNetwork

    convolutional_network = ConvolutionalNetwork()
    convolutional_network.load_state_dict(torch.load(weights_path, weights_only=False))
    convolutional_network.eval()
    
    image = Image.open(image_path).convert("L")
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ]) 
    input_tensor = transform(image).unsqueeze(0) #converte l'input da (channels, height, width) a (1, channels, height, width)
    
    with torch.no_grad():
        predict = convolutional_network(input_tensor)

    probabilities = F.softmax(predict, dim=1)
    print(probabilities)
    predicted_class = probabilities.argmax(dim=1)
    print(f"Predicted Class: {predicted_class.item()}")

    
if __name__ == "__main__":
    train()
    predict(
        weights_path='convolutional_weights.pth',
        image_path='/Users/nathanmonzani/Downloads/test.png')