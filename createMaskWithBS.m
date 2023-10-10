function mask = createMaskWithBS(v, meanScene, desvScene, times)
    for i=1 : size(v,3) %for each frame
        threshold = desvScene*times;
        mask(:,:,i) = abs(single(v(:,:,i)) - meanScene) > threshold;
    end
end