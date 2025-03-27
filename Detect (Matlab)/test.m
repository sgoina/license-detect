% 取得資料夾中所有的jpg文件
jpgFiles = dir(fullfile('.\img\', '*.jpg'));
output_txt = fopen('output.txt','w');

% 遍歷每個jpg文件
for i = 1:length(jpgFiles)
    % 建立目前文件的完整路徑
    currentFilePath = fullfile('.\img\', jpgFiles(i).name);
    fprintf(output_txt, '%s\n', jpgFiles(i).name);
    img = imread(currentFilePath);
    
    %找車牌
    find_card(img, output_txt);
end

function find_card(img, output_txt)
    %灰度轉換
    gray_img = rgb2gray(img);
    
    %圖像平滑處理
    smooth_img = medfilt2(gray_img);
    
    %找直的邊緣
    px1 = [-1, 0, 1;-1, 0, 1;-1, 0, 1];
    px2 = [1, 0, -1;1, 0, -1;1, 0, -1];
    sx1 = imfilter(smooth_img, px1);
    sx2 = imfilter(smooth_img, px2);
    sx = (sx1 + sx2) / 2;
    
    %二值化
    binary_img = imbinarize(sx);
    
    %關閉後開啟
    arr_b = ones([3,20]);
    temp = imclose(binary_img, arr_b);
    temp2 = imopen(temp, arr_b);
    
    %連通區域標記
    labeled_img = bwlabel(temp2,4);
    % figure, imshow(labeled_img)
    
    %選擇車牌區域
    stats = regionprops(labeled_img, 'Area', 'BoundingBox');
    minArea = 2500;  %最小面積
    maxArea = 30000;  %最大面積
    minAspectRatio = 1.5;  %最小長寬比
    maxAspectRatio = 4.5;  %最大長寬比
    
    %找出車牌框
    card_arr = [];
    for i = 1:length(stats)
        aspectRatio = stats(i).BoundingBox(3) / stats(i).BoundingBox(4);
        if (stats(i).Area > minArea) && (stats(i).Area < maxArea) && (aspectRatio > minAspectRatio) && (aspectRatio < maxAspectRatio)
            card_arr(end + 1) = i;
        end
    end
    %字元切割
    for now_card = 1:length(card_arr)
        s = stats(card_arr(now_card));
        [row, col, color] = size(img);
        if (round(s.BoundingBox(2)) + s.BoundingBox(4) > row || round(s.BoundingBox(1)) + s.BoundingBox(3) > col)
            continue;
        end
        card = img(round(s.BoundingBox(2)):round(s.BoundingBox(2)) + s.BoundingBox(4), round(s.BoundingBox(1)):round(s.BoundingBox(1)) + s.BoundingBox(3), :);
        card = rgb2gray(card);
        bin_card = imbinarize(card);
        arr_col = [0 1 0; 0 1 0;0 1 0];
        arr_row = [0 0 0; 1 1 1;0 0 0];
        bin_card = imclose(bin_card, arr_row);
        bin_card = imclose(bin_card, arr_col);
        [card_row, card_col] = size(bin_card);
        labeled_card = bwlabel(~bin_card, 4);
        % figure, imshow(labeled_card) ,impixelinfo
        card_left = card_col;
        for i = 1:card_row
            for j = 2:card_left
                if (bin_card(i, j-1) == 1 && bin_card(i,j) == 0)
                    card_left = min(card_left, j);
                end
            end
        end
        card_right = 1;
        for i = 1:card_row
            for j = card_col - 1:-1:card_right
                if (bin_card(i, j+1) == 1 && bin_card(i,j) == 0)
                    card_right = max(card_right, j);
                end
            end
        end
        bin_card = bin_card(1:card_row,card_left:card_right);

        %連通區域標記
        labeled_card = bwlabel(~bin_card, 4);
        % figure, imshow(labeled_card) ,impixelinfo

        %找出字元
        stats_card = regionprops(labeled_card, 'Area', 'BoundingBox');
        card_minRatio = 1;
        card_height = 20;
        word_arr = [];

        for i = 1:length(stats_card)
            aspectRatio = stats_card(i).BoundingBox(4) / stats_card(i).BoundingBox(3);
            if (stats_card(i).BoundingBox(4) > card_height && aspectRatio > card_minRatio)
                word_arr(end + 1) = i;
            end
        end

        %刪除字元數小於4的圖
        if (length(word_arr) <= 4)
            continue;
        end
        if (length(word_arr) >= 7)
            % word_height = [];
            % for i = 1:length(word_arr)
            %     word_height(end + 1) = stats_card(word_arr(i)).BoundingBox(4);
            % end
            % mean_height = mean(word_height);
            % std_height = std(word_height);
            % new_word_arr = [];
            % for i = 1:length(word_arr)
            %     if (word_height(i) <= mean_height + std_height && word_height(i) >= mean_height - std_height)
            %         new_word_arr(end+1) = word_arr(i);
            %     end
            % end
            % word_arr = new_word_arr;
            leftup_x = [];
            leftup_y = [];
            for i = 1:length(word_arr)
                leftup_x(i) = stats_card(word_arr(i)).BoundingBox(1);
                leftup_y(i) = stats_card(word_arr(i)).BoundingBox(2);
            end
            coefficients = polyfit(leftup_x, leftup_y, 1);
            distance = [];
            for i = 1:length(word_arr)
                distance(i) = abs(leftup_x(i) * coefficients(1) - leftup_y(i) + coefficients(2));
                distance(i) = distance(i) / sqrt(coefficients(1) * coefficients(1) + 1);
            end
            std_distance = std(distance);
            mean_distance = mean(distance);
            if (length(word_arr) > 7 || (length(word_arr) == 7 && std_distance > 3))
                new_word_arr = [];
                for i = 1:length(word_arr)
                    if (distance(i) <= mean_distance + std_distance)
                        new_word_arr(end+1) = word_arr(i);
                    end
                end
                word_arr = new_word_arr;
            end
        end
        % figure, imshow(card)
        fprintf(output_txt, '%d\n', length(word_arr));


        % 輸出車牌框`
        for i = 1:length(word_arr)
            stats_card(word_arr(i)).BoundingBox(1) = round(stats_card(word_arr(i)).BoundingBox(1) + s.BoundingBox(1)+card_left-1);
            stats_card(word_arr(i)).BoundingBox(2) = round(stats_card(word_arr(i)).BoundingBox(2) + s.BoundingBox(2)-1);
            fprintf(output_txt,'%d %d %d %d\n',stats_card(word_arr(i)).BoundingBox);
        end
        figure;
        imshow(img);
        hold on;
        for i = 1:length(word_arr)
            rectangle('Position', stats_card(word_arr(i)).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 2);
        end
        impixelinfo,
        hold off;
    end
end