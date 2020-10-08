def trainModel(documentDF: DataFrame): Word2VecModel = {
    var word2vecModel: Word2VecModel = null
    val name = "Word2VecModel"
    if (Globals.PRODUCTION) {
      try {
        word2vecModel = Word2VecModel.load(name)
      } catch {
        case e: Exception => throw new Exception(e + "\nMust pre-train models first, set PRODUCTION = false");
      }
    }
    else {
      val word2Vec = new Word2Vec()
        .setInputCol("FilteredSentimentText")
        .setOutputCol("features")
        .setVectorSize(50)
        .setMinCount(5)
        .setSeed(Globals.SEED)
      word2vecModel = word2Vec.fit(documentDF)
      word2vecModel.write.overwrite().save(name)
    }
        word2vecModel
  }

  def loadModel(): Word2VecModel = {
    var word2vecModel: Word2VecModel = null
    val name = "Word2VecModel"
    try {
      word2vecModel = Word2VecModel.load(name)
    } catch {
      case e: Exception => throw new Exception(e + "\nMust pre-train models first, set PRODUCTION = false");
    }
    word2vecModel
  }
