function colornorm(sourcename, targetname)
    basepath = 'D:\ZF_Histo_image\image\';
    start_spams; 
    nstains = 2; 
    lambda = 0.02; 

    types = ["_Default_Extended" "-jizhi" "-tumor-1" "-tumor-2" "-tumor-3" "-tumor-4"];
    len = length(types);
    totalHivs = [];
    totalWit = [];
    totalHivt = [];
    sources = [];
    targets = [];
    
    parpool; 
    for i = 1:len
        source = imread(strcat(basepath, sourcename, types{i}, ".tif"));
        source = source(1:1600, 1:1600, :);
        target = imread(strcat(basepath, targetname, types{i}, ".tif"));
        target = target(1:1600, 1:1600, :);
        [Wis, His, Hivs, stainss] = Faststainsep(source, nstains, lambda); 
        [Wit, Hit, Hivt, stainst] = Faststainsep(target, nstains, lambda); 
        totalHivs = cat(3, totalHivs, Hivs);
        totalHivt = cat(1, totalHivt, Hivt); % Hivt = n x r
        totalWit = cat(3, totalWit, Wit);
        sources = cat(4, sources, source);
        targets = cat(4, targets, target);
    end
    delete('gcp')
    
    Hta_Rmax  =  prctile(totalHivt, 99); 
    totalWit = mean(totalWit, 3);
    totalWit = totalWit ./ sqrt(sum(totalWit .^ 2, 1));
    fig = tight_plot(len, 3, [.01 .03], [.1 .01], [.01 .01]);
    
    for i = 1:len
        Hivs = totalHivs(:, :, i);
        source = sources(:, :, :, i);
        target = targets(:, :, :, i);
        Hso_Rmax  =  prctile(Hivs, 99); 
        normfac = (Hta_Rmax ./ Hso_Rmax); 
        Hsonorm  =  Hivs .* repmat(normfac, size(Hivs, 1), 1); 

        imgsource_norm  =  totalWit * Hsonorm'; 
        rows = size(source, 1); cols = size(source, 2); 
        sourcenorm = uint8(255 * exp(-reshape(imgsource_norm', rows, cols, 3)));
        axes(fig(3 * i - 2)); imshow(source); ylabel(types{i});
        if i == len
            xlabel('source series'); 
        end
        axes(fig(3 * i - 1)); imshow(target); 
        if i == len
            xlabel('target series'); 
        end
        axes(fig(3 * i)); imshow(sourcenorm); 
        if i == len
            xlabel('normalized source series'); 
        end
    end
end