#%%
import matplotlib.pyplot as plt
import tensorflow as tf
from transformer import positional_encoding


#%%
pos_encoding = positional_encoding(length=2048, depth=512)
print(pos_encoding.shape)

#%% Plot the dimensions.
plt.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()

#%% By definition these vectors align well with nearby vectors 
#   along the position axis. Below the position encoding vectors 
#   are normalized and the vector from position 1000 is compared, 
#   by dot-product, to all the others
pos_encoding/=tf.norm(pos_encoding, axis=1, keepdims=True)
p = pos_encoding[1000]
dots = tf.einsum('pd,d -> p', pos_encoding, p)
plt.subplot(2,1,1)
plt.plot(dots)
plt.ylim([0,1])
plt.plot([950, 950, float('nan'), 1050, 1050],
         [0,1,float('nan'),0,1], color='k', label='Zoom')
plt.legend()
plt.subplot(2,1,2)
plt.plot(dots)
plt.xlim([950, 1050])
plt.ylim([0,1])


# %%
