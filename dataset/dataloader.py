import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

# import sys
# sys.path.append('/ssd6/Roy/Xception_copy/')
from dataset.dataset_fast import FaceForensics
from dataset.dataset_celebDF import CelebDF_fewimage, CelebDF_JYUNYI

def collate_fn(batch):
    imgs = [item['image'] for item in batch if item['image'] is not None]
    targets = [item['label'] for item in batch if item['image'] is not None]
    filenames = [item['filename'] for item in batch if item['image'] is not None]
    imgs = torch.stack(imgs)
    targets = torch.stack(targets)
    return {'image': imgs, 'label': targets, 'filename': filenames}

def get_loader(args):
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    train_data = FaceForensics(args.root_dir, args.train_file_path, transform = transform)
    train_loader = DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)

    val_data = FaceForensics(args.root_dir, args.val_file_path, transform = transform)
    val_loader = DataLoader(dataset = val_data, batch_size = args.batch_size, shuffle=True, collate_fn=collate_fn)

    test_data = FaceForensics(args.root_dir, args.test_file_path, img_batch = args.test_img_batch, transform = transform)
    test_loader = DataLoader(dataset = test_data, batch_size = args.test_img_batch, shuffle = False, collate_fn=collate_fn)
    
    return train_data, val_data, test_data, train_loader, val_loader, test_loader

def get_loader_celebDF_fewimage(args):
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    train_data = CelebDF_fewimage(args.root_dir, args.train_file_path, transform = transform)
    train_loader = DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)

    val_data = CelebDF_fewimage(args.root_dir, args.val_file_path, transform = transform)
    val_loader = DataLoader(dataset = val_data, batch_size = args.batch_size, shuffle=True, collate_fn=collate_fn)

    test_data = CelebDF_fewimage(args.root_dir, args.test_file_path, img_batch = args.test_img_batch, transform = transform)
    test_loader = DataLoader(dataset = test_data, batch_size = args.test_img_batch, shuffle = False, collate_fn=collate_fn)
    
    return train_data, val_data, test_data, train_loader, val_loader, test_loader



def get_loader_celebDF_JYUNYI(args):
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    train_data = CelebDF_JYUNYI(args.root_dir, args.train_file_path, transform = transform)
    train_loader = DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)

    val_data = CelebDF_JYUNYI(args.root_dir, args.val_file_path, transform = transform)
    val_loader = DataLoader(dataset = val_data, batch_size = args.batch_size, shuffle=True, collate_fn=collate_fn)

    # test_data = CelebDF_JYUNYI(args.root_dir, args.test_file_path, img_batch = args.test_img_batch, transform = transform)
    # test_loader = DataLoader(dataset = test_data, batch_size = args.test_img_batch, shuffle = False, collate_fn=collate_fn)
    
    # return train_data, val_data, test_data, train_loader, val_loader, test_loader

    return train_data, val_data, train_loader, val_loader