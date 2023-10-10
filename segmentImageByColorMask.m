function imContainer = segmentImageByColorMask(v, maskColorDepth, i)
    imAux = v(:,:,i);
    imContainer = imAux*NaN;
    imContainer(maskColorDepth(:,:,i)) = imAux(maskColorDepth(:,:,i));
end