import torch
from torch import nn

class LSTM_model(nn.Module):
    def __init__(self, num_classes=3, input_features=22):
        super(LSTM_model, self).__init__()
        # Make sure input_features is at least 3 for the Conv1d to work properly
        if input_features < 3:
            input_features = 3
            print("Warning: input_features increased to 3 for Conv1d to work")
            
        self.input_features = input_features
        self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1,
                              padding=1)  # Changed padding to 1 to maintain sequence length
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=2,
                              batch_first=True)
        self.fc = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # (n, 1, 22)
        x = self.conv(x)  # (n, 16, 20)  # Note: size is reduced due to kernel size 3
        
        x = x.permute(0, 2, 1)  # (n, 20, 16)
        
        lstm_out, _ = self.lstm(x)  # (n, 20, 32)
        
        output = self.fc(lstm_out)  # (n, 20, num_classes)
        output = self.softmax(output)  # Apply softmax for multi-class classification
        
        # Get the output from the last timestep for classification
        output_class = output[:, -1, :]  # (n, num_classes)
        
        return output_class

if __name__ == "__main__":
    model = LSTM_model()
    # Example input from the CSV format (22 features)
    example_input = torch.tensor([[2048, 2048, 0.000164, 0.000423, 0.002709, 8.11E-06, 
                                  0.000484, 6.19E-06, 4.24E-07, 0.000188, 2.22E-08, 
                                  2.79E-08, 6.31E-07, 7.48E-10, 1.81E-08, 1.43E-09, 
                                  3.64E-11, 1.41E-08, 3.73E-12, 1.99E-12, 1.51E-10, 
                                  7.94E-14]], dtype=torch.float32)

    output_class = model(example_input)
    print("output_class size", output_class.size())
    print("output_class", output_class)
    print("Predicted class:", torch.argmax(output_class, dim=1) + 1)  # Adding 1 since class labels are 1, 2, 3