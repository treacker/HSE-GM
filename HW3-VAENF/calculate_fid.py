import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def tensorify(lst):
    """
    List must be nested list of tensors (with no varying lengths within a dimension).
    Nested list of nested lengths [D1, D2, ... DN] -> tensor([D1, D2, ..., DN)

    :return: nested list D
    """
    # base case, if the current list is not nested anymore, make it into tensor
    if type(lst[0]) != list:
        if type(lst) == torch.Tensor:
            return lst
        elif type(lst[0]) == torch.Tensor:
            return torch.stack(lst, dim=0)
        else:  # if the elements of lst are floats or something like that
            return torch.tensor(lst)
    current_dimension_i = len(lst)
    for d_i in range(current_dimension_i):
        tensor = tensorify(lst[d_i])
        lst[d_i] = tensor
    # end of loop lst[d_i] = tensor([D_i, ... D_0])
    tensor_lst = torch.stack(lst, dim=0)
    return tensor_lst

def calculate_activation_statistics(image, model, classifier):
    classifier.eval()
    model.eval()
    device = next(model.parameters()).device
    
    # Здесь ожидается что вы пройдете по данным из даталоадера и соберете активации классификатора для реальных и сгенерированных данных
    # После этого посчитаете по ним среднее и ковариацию, по которым посчитаете frechet distance
    # В целом все как в подсчете оригинального FID, но с вашей кастомной моделью классификации
    # note: не забывайте на каком девайсе у вас тензоры 
    # note2: не забывайте делать .detach()
    
    original, reconstructed = torch.tensor([]).to(device), torch.tensor([]).to(device)
    
    with torch.no_grad():
        
        image = image.to(device)
        
        new_image = model.sample(len(image), image.size(3))
        new_image = torch.clamp(new_image, -1, 1)
        real_activs = classifier(image)
        real_activs = tensorify(real_activs).reshape((image.shape[0], -1))
        fake_activs = classifier(new_image)
        fake_activs = tensorify(fake_activs).reshape((image.shape[0], -1))
        
        original = torch.cat((original, real_activs), 0)
        reconstructed = torch.cat((reconstructed, fake_activs), 0)

    m1 = original.mean(dim=0)
    m2 = reconstructed.mean(dim=0)

    s1 = np.cov(original.to('cpu').detach(), rowvar=False)
    s2 = np.cov(reconstructed.to('cpu').detach(), rowvar=False)
    
    return m1.to('cpu').detach(), s1, m2.to('cpu').detach(), s2


@torch.no_grad()
def calculate_fid(image, model, classifier):
    
    m1, s1, m2, s2 = calculate_activation_statistics(image, model, classifier)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value.item()
