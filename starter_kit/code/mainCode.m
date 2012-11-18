clear all;
load ../data/music_dataset.mat


%test my function, need to revise later
%train = train(1:1000);

[Xt_lyrics] = make_lyrics_sparse(train, vocab);
[Xq_lyrics] = make_lyrics_sparse(quiz, vocab);

Yt = zeros(numel(train), 1);

for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

%Xt_audio = make_audio(train);
%Xq_audio = make_audio(quiz);

%partition for crossing validation and test
partition = make_part(Xt_lyrics);

%seperate the training set, cross validation set, and test set?

training_set = Xt_lyrics(partition == 1,:);
training_label = Yt(partition == 1,:)

for i = 2:6,
    training_set = [training_set; Xt_lyrics(partition == i,:)];
    training_label = [training_label; Yt(partition == i,:)];
end

cross_set = Xt_lyrics(partition == 7,:);
cross_label = Yt(partition == 7,:);
cross_set = [cross_set;Xt_lyrics(partition == 8,:)];
cross_label = [cross_label; Yt(partition == 8,:)];

test_set = Xt_lyrics(partition == 9,:);
test_label = Yt(partition == 9,:);
test_set = [test_set;Xt_lyrics(partition == 10,:)];
test_label = [test_label; Yt(partition == 10,:)];

%use the logistic regression to train the training data

w = zeros(10, size(training_set,2) + 1);
binary_label = double(bsxfun(@eq,training_label,1:10));
%training for every label, we will get ten w
for i = 3:10,
   % training_label(training_label ~= i) = 0;
    [w(i,:),obj{i},~] = lr_train(training_set,binary_label(:,i),0.01,'max_iter',100,'step_size',0.0001);
end

save w.mat w
save obj
