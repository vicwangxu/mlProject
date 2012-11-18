%simple algorithm, just revise the example code

clear all;
load ../data/music_dataset.mat


%cross validation

%tr = train(1:8000);
%test = train(8001:9000);

cross_error = zeros(10,1);


[Xt_lyrics] = make_lyrics_sparse(train, vocab);
%[Xq_lyrics] = make_lyrics_sparse(test, vocab);
Yt = zeros(numel(train), 1);

for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

for round = 1:10,

partition = make_part(Xt_lyrics,10);

%seperate the training set, cross validation set, and test set?

training_set = Xt_lyrics(partition == 1,:);
training_label = Yt(partition == 1,:);


for i = 2:8,
    training_set = [training_set; Xt_lyrics(partition == i,:)];
    training_label = [training_label; Yt(partition == i,:)];
end

cross_set = Xt_lyrics(partition == 9,:);
cross_label = Yt(partition == 9,:);

for i = 10
       cross_set = [cross_set; Xt_lyrics(partition == i,:)];
       cross_label = [cross_label; Yt(partition == i,:)];
end
%Ycross = zeros(numel(cross_label),1);


Xt_audio = 0;% make_audio(training_set);
Xq_audio = 0;%make_audio(quiz);

%% Run algorithm
ranks = predict_genre(training_set, cross_set, ...
                      Xt_audio, Xq_audio, ...
                      training_label);

rank_error = rank_loss(ranks, cross_label);

cross_error(round) = rank_error;
end
                  
%% Save results to a text file for submission
%save('-ascii', 'submit.txt', 'ranks');
