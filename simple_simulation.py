import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from scipy.spatial.distance import cdist
from os import system
from sklearn.neighbors import KernelDensity

# Kolory zależne od prędkości
NDIM = 2

# Map parameters
edge = 10
sigma = .1
alpha = .75
nts = 1/7 # niepokój
n_birds = 2
n_hidden = 10
n_epochs = 1024
history = 92
bratio = 20
levels = 8

representation = np.zeros((NDIM + n_birds * 2))
interaction = np.zeros((NDIM + 1))

bird = MLPRegressor(hidden_layer_sizes=(representation.shape[0], 
                                         n_hidden,
                                         n_hidden,
                                         #n_hidden,
                                         interaction.shape[0]),
                    random_state=None,
                    solver='lbfgs')

birds = [clone(bird) for i in range(n_birds)]
representations = np.random.uniform(-edge, edge, (n_birds, *representation.shape))
interactions = np.random.normal(0, sigma, (n_birds, *interaction.shape))

szlak = []
cszlak = []

for i in range(n_epochs):
    print(i, representations.shape, interactions.shape)
    # Move
    representations[:,:2] += interactions[:, :2] 
    
    # Update representation with distance awareness
    distances = cdist(representations[:,:2], representations[:,:2])
    distances = distances[distances[:, 0].argsort()]
    representations[:,2:(2+n_birds)] = distances
    #representations[:,2:] = np.max(representations[:,2:]) - representations[:,2:]
    
    # Update representation with interaction awareness
    representations[:,(2+n_birds):] = interactions[:,2:]
    
    # Teleport at the edge
    representations[representations>edge] -= 2 * edge
    representations[representations<-edge] += 2 * edge    
    
    #representations[representations<-edge] += 2 * edge

    
    # Fit to situation
    for bid, bird in enumerate(birds):
        bird.fit(representations[bid][None, :], 
                 interactions[bid][None, :])
        
        if i % n_birds == n_birds - 1:
            bird.fit(representations, interactions)
        
    
    # Predict movement
    a = interactions * alpha 
    b = np.array([bird.predict(representations[bid][None, :]) 
                  for bid, bird in enumerate(birds)]) * (1-alpha)
    b = b.squeeze()
    interactions = a + b
    interactions += np.random.normal(0, sigma, interactions.shape) * nts
    
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    #ax = ax.ravel()
    
    colors = np.sum(np.abs(interactions[:,:2]), axis=1)
    colors -= np.min(colors)
    colors /= np.max(colors)
    #colors[:, :2] /= 4
    
    
    aa = ax
    #ab = ax[1]
    #ac = ax[3]
    #ad = ax[2]
    
    szlak.append(representations[:,:2].copy())
    cszlak.append(colors.copy())
    
    if len(szlak) > history:
        szlak = szlak[1:]
    
    if len(cszlak) > history:
        cszlak = cszlak[1:]
        
    conszlak = np.concatenate(szlak)
    
    kde = KernelDensity(kernel='gaussian', bandwidth=edge/bratio).fit(conszlak)
    
    a, b = np.meshgrid(np.linspace(-edge, edge, 100), np.linspace(-edge, edge, 100))
    z = np.exp(kde.score_samples(np.array([a.ravel(), b.ravel()]).T))
            
    #samples = kde.sample(1000)#n_birds * 32)

    #kde2 = KernelDensity(kernel='gaussian').fit(samples)
    #a, b = np.meshgrid(np.linspace(-edge, edge, 100), np.linspace(-edge, edge, 100))
    #z2 = np.exp(kde2.score_samples(np.array([a.ravel(), b.ravel()]).T))

    
    aa.contourf(a, b, z.reshape(a.shape), cmap='jet', levels=levels, alpha = .25,zorder=-101)
    aa.contour(a, b, z.reshape(a.shape), cmap='gray', alpha=.25, levels=levels,zorder=-102)

    
    #ad.contourf(a, b, z2.reshape(a.shape), cmap='turbo', levels=16)
    
    #ac.scatter(*samples.T, c='black', s=5, alpha=.5)
        
    #print(z.shape)
    #r = z.reshape(100,100)
    #g = z2.reshape(100,100)
    #b = r*.5 + g*.5
    
    #rgb = np.concatenate((r[:,:,None],
    #                      g[:,:,None],
    #                      b[:,:,None]), axis=2)
    #rgb = np.exp(rgb)
    #rgb -= np.min(rgb)
    #rgb /= np.max(rgb)
    
    #rgb = rgb
    
    #ac.imshow(g-r, origin='lower', cmap='terrain') 
    #ac.contour(a, b, z2.reshape(a.shape), color='white', levels=16)
    
    aa.scatter(*np.concatenate(szlak).T, s=2, alpha=.1, 
               c='black',#np.concatenate(cszlak), 
               marker='o')

    #aa.scatter(*representations[:,:2].T, s=25, alpha=1, c=colors[:,-1], 
    #           marker='x', cmap='Set1')

    aa.scatter(*representations[:,:2].T, s=250, alpha=1, c='white', 
               marker='o')
    aa.scatter(*representations[:,:2].T, s=150, alpha=1, c=colors, 
               marker='2', cmap='brg')

    #aa.scatter(*representations[:,:2].T, s=250, alpha=1, c='tomato', 
    #           marker='o')
    
    #aa.scatter(*representations[:,:2].T, s=50, alpha=1, c=colors, 
    #           marker='o')
    
    #aa.scatter(*representations[:,:2].T, s=100, alpha=.25, c='black', 
    #           marker='x')
    
    #aa.plot(*representations[:,:2].T, c='k', alpha=.1)
    
    aa.grid(ls=":")
    aa.set_xlim(-edge*1.1, edge*1.1)
    aa.set_ylim(-edge*1.1, edge*1.1)

    #ac.grid(ls=":")
    #ac.set_xlim(-edge*1.1, edge*1.1)
    #ac.set_ylim(-edge*1.1, edge*1.1)
    
    for az in [aa]:
        az.spines['top'].set_visible(False)
        az.spines['right'].set_visible(False)
        az.spines['left'].set_visible(False)
        az.spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Time: {}'.format(i))
    plt.savefig('foo.png')
    plt.savefig('frames/%04i.png' % i)
    plt.close()
    
    system('cp foo.png bar.png')