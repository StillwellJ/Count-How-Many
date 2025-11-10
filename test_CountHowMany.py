from CountHowMany import counter

def test_model():
    assert type(counter("test.jpg")) == int

def test_model_2():
    assert type(counter("test2.jpg")) == int

def test_model_3():
    assert type(counter("test3.jpg")) == int