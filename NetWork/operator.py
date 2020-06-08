import torch


def l2_norm(x, ratio=1.0, axis=1):
    norm = torch.unsqueeze(torch.clamp(torch.norm(x, 2, axis), min=1e-6), axis)
    x = x / norm * ratio
    return x


def normalize_coordinates(coords, h, w):
    h = h - 1
    w = w - 1
    coords = coords.clone().detach()
    coords[:, :, 0] -= w / 2
    coords[:, :, 1] -= h / 2
    coords[:, :, 0] /= w / 2
    coords[:, :, 1] /= h / 2
    return coords


def scale_rotate_offset_dist(feats0, feats1, scale_offset, rotate_offset, max_sn, max_rn):
    '''

    :param feats0:          b,n,ssn,srn,f
    :param feats1:          b,n,ssn,srn,f
    :param scale_offset:    b,n
    :param rotate_offset:   b,n
    :param max_sn:
    :param max_rn:
    :return: dist: b,n
        scale_offset<0: img0 0:ssn+scale_offset corresponds to img1 -scale_offset:ssn
        scale_offset>0: img0 scale_offset:ssn corresponds to img1 0:ssn-scale_offset
        rotate_offset<0: img0 0:srn+rotate_offset corresponds to img1 -rotate_offset:srn
        rotate_offset>0: img0 rotate_offset:srn corresponds to img1 0:srn-rotate_offset
    '''
    b, n, ssn, srn, f = feats0.shape
    dist = torch.zeros([b, n], dtype=torch.float32, device=feats0.device)
    for si in range(-max_sn, max_sn + 1):
        for ri in range(-max_rn, max_rn + 1):
            mask = (scale_offset == si) & (rotate_offset == ri)
            if torch.sum(mask) == 0: continue
            cfeats0 = feats0[mask]  # n, ssn, srn, f
            cfeats1 = feats1[mask]  # n, ssn, srn, f
            if si < 0:
                cfeats0 = cfeats0[:, :ssn + si]
                cfeats1 = cfeats1[:, -si:]
            elif si > 0:
                cfeats0 = cfeats0[:, si:]
                cfeats1 = cfeats1[:, :ssn - si]
            else:
                cfeats0 = cfeats0
                cfeats1 = cfeats1

            if ri < 0:
                cfeats0 = cfeats0[:, :, :srn + ri]
                cfeats1 = cfeats1[:, :, -ri:]
            elif ri > 0:
                cfeats0 = cfeats0[:, :, ri:]
                cfeats1 = cfeats1[:, :, :srn - ri]
            else:
                cfeats0 = cfeats0
                cfeats1 = cfeats1

            cdist = torch.norm(cfeats0 - cfeats1, 2, 3)  # n,
            n, csn, crn = cdist.shape
            cdist = torch.mean(cdist.reshape(n, csn * crn), 1)
            dist[mask] = dist[mask] + cdist

    return dist
