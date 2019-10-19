function acc = CMC(dist,Rank)

    [~,index] = sort(dist,2);
    rank = zeros(size(dist,1),1);

    for i = 1:size(index,1)
        rank(i) = find(index(i,:) == i);
    end

    acc = zeros(1,Rank);
    for i = 1:Rank
        acc(i) = 100*mean(rank<=i);
    end

    plot(1:Rank, acc,'LineWidth',3);
    grid on;
    xlabel('Rank')
    ylabel('Accuracy')
    title('CMC curve')

end
