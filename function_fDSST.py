def get_scale_subwindow(img, pos, base_target_sz, scaleFactors, scale_model_sz):
    nScales = len(scaleFactors)
    
    for s in range(nScales):
        patch_sz = np.floor(base_target_sz * scaleFactors[s])
        
        xs = np.floor(pos[1]) + range(1,int(patch_sz[1])+1) - np.floor(patch_sz[1]//2)
        ys = np.floor(pos[0]) + range(1,int(patch_sz[0])+1) - np.floor(patch_sz[0]//2)
        
        xs[xs < 1] = 1
        ys[ys < 1] = 1
        xs[xs > img.shape[1]] = img.shape[1]
        ys[ys > img.shape[0]] = img.shape[0]
    
        im_patch = img[int(ys[0])-1:int(ys[-1]), int(xs[0])-1:int(xs[-1]), :]
        im_patch_resized = cv2.resize(im_patch,tuple(scale_model_sz[::-1].astype(int)))
        
        M = np.zeros(im_patch_resized.shape[:2], dtype='float32')
        O = np.zeros(im_patch_resized.shape[:2], dtype='float32')
        fhog.gradientMag(im_patch_resized.astype(np.float32),M,O)
        H = np.zeros([im_patch_resized.shape[0]//4, im_patch_resized.shape[1]//4, 32], dtype='float32')
        fhog.gradientHist(M,O,H)
        
        if s == 0:
            dim_scale = H.shape[0]*H.shape[1]*31
            out_pca = np.zeros([dim_scale, nScales], dtype='float32')
            
        out_pca[:,s] = H[:,:,:31].reshape(-1, order="F")

    return out_pca
    
def feature_projection_scale(xs_pca, projection_matrix, cos_window):
    tmp = np.matrix(projection_matrix)*np.matrix(xs_pca)
    tmp = np.array(tmp)
    return tmp*np.tile(cos_window, (tmp.shape[0], 1))
    
    
def resizeDFT(inputdft, desiredLen):
    
    len_ = len(inputdft)
    minsz = min(len_, desiredLen)
    
    scaling = desiredLen // len_
    
    
    resizeddft = np.zeros([1,int(desiredLen)], dtype='float32')+np.zeros([1,int(desiredLen)], dtype='float32')*1j
    
    mids = np.ceil(minsz/2.0)
    mide = np.floor((minsz-1)/2.0) - 1
    
    resizeddft[0,:int(mids)] = scaling * inputdft[:int(mids)]
    resizeddft[0,-1-int(mide):] = scaling * inputdft[-1-int(mide):]
    
    return  resizeddft


"""first frame"""
lambda_ = 1e-2
padding = 2.0
nScales = 17.0
scale_step = 1.02
nScalesInterp = 33.0
interp_factor = 0.025
scale_model_factor = 1.0
scale_sigma_factor = 1/16.0
scale_model_max_area = 512.0

pos_s = (bbox[:2] + bbox[2:]/2.0)[::-1]
target_sz = bbox[2:][::-1]
init_target_sz = bbox[2:][::-1]

currentScaleFactor = 1.0

base_target_sz = target_sz / currentScaleFactor

sz = np.floor( base_target_sz * (1 + padding ))


scale_sigma = nScalesInterp * scale_sigma_factor

scale_exp = np.array(range(-np.floor((nScales-1)/2).astype(int),np.ceil((nScales-1)/2).astype(int)+1)) * nScalesInterp/nScales
scale_exp_shift = np.roll(scale_exp, -np.floor((nScales-1)/2).astype(int))

interp_scale_exp = np.array(range(-np.floor((nScalesInterp-1)/2).astype(int), np.ceil((nScalesInterp-1)/2).astype(int)+1))
interp_scale_exp_shift = np.roll(interp_scale_exp, -np.floor((nScalesInterp-1)/2).astype(int))

scaleSizeFactors = np.power(scale_step,scale_exp)
interpScaleFactors = np.power(scale_step,interp_scale_exp_shift)

ys = np.exp(-0.5*np.power(scale_exp_shift,2) / scale_sigma**2)
ysf = fft(ys)
scale_window = np.hanning(len(ysf))


if scale_model_factor**2 * np.prod(init_target_sz) > scale_model_max_area:
    scale_model_factor = np.sqrt(scale_model_max_area/np.prod(init_target_sz))

scale_model_sz = np.floor(init_target_sz * scale_model_factor)


min_scale_factor = np.power( scale_step, np.ceil(np.log(np.max(5/sz)) / np.log(scale_step)) )
max_scale_factor = np.power( scale_step, np.floor(np.log(np.min(img.shape[:2] / base_target_sz)) / np.log(scale_step)) )

max_scale_dim = 1
s_num_compressed_dim = len(scaleSizeFactors)

##############
xs_pca = get_scale_subwindow(img, pos_s, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz)
    
s_num = xs_pca.copy()
bigY = s_num.copy()
bigY_den = xs_pca.copy()

scale_basis,_ = np.linalg.qr(bigY)
scale_basis_den,_ = np.linalg.qr(bigY_den)

scale_basis = scale_basis.T

#create the filter update coefficients
tmp = feature_projection_scale(s_num,scale_basis,scale_window)
sf_proj = fft(tmp,tmp.shape[1],1)   
sf_num = np.tile(ysf, (tmp.shape[0], 1))*conj(sf_proj)
            
xs = feature_projection_scale(xs_pca,scale_basis_den.T,scale_window)
xsf = fft(xs,xs.shape[1],1)           
new_sf_den = np.sum(xsf * conj(xsf),0)

sf_den = new_sf_den.copy()


"""estimate scale"""
pos_s[0] = pos[1]
pos_s[1] = pos[0]
xs_pca = get_scale_subwindow(img, pos_s, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz)

xs = feature_projection_scale(xs_pca,scale_basis,scale_window)
xsf = fft(xs,xs.shape[1],1)

scale_responsef = np.sum(sf_num * xsf, 0) / (sf_den + lambda_)


interp_scale_response = real(ifft( resizeDFT(scale_responsef, nScalesInterp)[0]))

recovered_scale_index = np.where(interp_scale_response == np.max(interp_scale_response))[0][0]

currentScaleFactor = currentScaleFactor * interpScaleFactors[recovered_scale_index]

if currentScaleFactor < min_scale_factor:
    currentScaleFactor = min_scale_factor
elif currentScaleFactor > max_scale_factor:
    currentScaleFactor = max_scale_factor
print(currentScaleFactor)
# update

xs_pca = get_scale_subwindow(img, pos_s, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz)
    
s_num = (1 - interp_factor) * s_num + interp_factor * xs_pca.copy()
bigY = s_num.copy()
bigY_den = xs_pca.copy()

scale_basis,_ = np.linalg.qr(bigY)
scale_basis_den,_ = np.linalg.qr(bigY_den)

scale_basis = scale_basis.T

#create the filter update coefficients
tmp = feature_projection_scale(s_num,scale_basis,scale_window)
sf_proj = fft(tmp,tmp.shape[1],1)   
sf_num = np.tile(ysf, (tmp.shape[0], 1))*conj(sf_proj)


xs = feature_projection_scale(xs_pca,scale_basis_den.T,scale_window)
xsf = fft(xs,xs.shape[1],1)

new_sf_den = np.sum(xsf * conj(xsf),0)

sf_den = (1 - interp_factor) * sf_den + interp_factor * new_sf_den.copy()

target_sz = np.floor(base_target_sz * currentScaleFactor)

bbox_est = np.array([pos[0]-target_sz[1]/2.0, pos[1]-target_sz[0]/2.0, target_sz[1], target_sz[0]])                    
bbox_est = np.round(bbox_est).astype(int) 
