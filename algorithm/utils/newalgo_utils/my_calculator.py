from benchmark.toolkits import ClassifyCalculator
import torch

class MyCalculator(ClassifyCalculator):
    @torch.no_grad()
    def test(self, model, data, device=None):
        """Metric = Accuracy"""
        feature_generator, classifier = model
        
        tdata = self.data_to_device(data, device)
        feature_generator = feature_generator.to(device).eval()
        classifier = classifier.to(device).eval()

        feature = feature_generator(tdata[0])
        outputs = classifier(feature)
        loss = self.lossfunc(outputs, tdata[-1])

        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
        return (1.0 * correct / len(tdata[1])).item(), loss.item()
