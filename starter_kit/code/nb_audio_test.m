clear all;
load ../data/music_dataset.mat


%cross validation

%tr = train(1:8000);
%test = train(8001:9000);

%test the program
%train = [train(:,1:2) train(:,4:30)];
%train = train(1:1000);

%train(:,(train(:,3) == -1)) = NaN; 
cross_error = zeros(10,1);
Xt_audio = make_audio(train);
Xt_audio = [Xt_audio(:,1:2) Xt_audio(:,4:30)];
%Xq_audio = make_audio(quiz);

%[Xt_lyrics] = make_lyrics_sparse(train, vocab);
%[Xq_lyrics] = make_lyrics_sparse(test, vocab);
Yt = zeros(numel(train), 1);

for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

%for round = 1:10,

partition = make_part(Xt_audio,10);

%seperate the training set, cross validation set, and test set?

training_set = Xt_audio(partition == 1,:);
training_label = Yt(partition == 1,:);


for i = 2:8,
    training_set = [training_set; Xt_audio(partition == i,:)];
    training_label = [training_label; Yt(partition == i,:)];
end

cross_set = Xt_audio(partition == 9,:);
cross_label = Yt(partition == 9,:);

for i = 10
       cross_set = [cross_set; Xt_audio(partition == i,:)];
       cross_label = [cross_label; Yt(partition == i,:)];
end
%Ycross = zeros(numel(cross_label),1);

nb = NaiveBayes.fit(training_set, training_label);
cpre = predict(nb,cross_set,'HandleMissing','on');

%Xt_audio = 0;% make_audio(training_set);
%Xq_audio = 0;%make_audio(quiz);

%% Run algorithm
%[ranks,info] = predict_genre(training_set, cross_set, ...
                      %0, 0, ...
                      %training_label);

%rank_error = rank_loss(ranks, cross_label);

%cross_error(1) = rank_error;