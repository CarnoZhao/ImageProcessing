using Images, PyPlot

function Demo_stainsep()
    I = imread("D:/Codes/ImageProcessing/stain_sep_color_norm/SNMF/images/he.png")
    nstains = 2
    lambda = 0.1
    Wi, Hi, Hiv, stains = stainsep(I, nstains, lambda)
    fig, axes = plt.subplots(1, 3)
    axes[1].imshow(I)
    axes[1].set_xlabel("Input")
    axes[2].imshow(stains[1])
    axes[2].set_xlabel("stain1")
    axes[3].imshow(stains[2])
    axes[3].set_xlabel("stain2")
    plt.savefig("D:/Codes/ImageProcessing/stain_sep_color_norm/SNMF/images/output.png")
end

function stainstep(I, nstains, lambda)
    
end