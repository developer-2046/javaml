package javaml.core;

import java.util.List;

public abstract class Optimizer{
    public abstract void update(List<Layer> layers);

}
