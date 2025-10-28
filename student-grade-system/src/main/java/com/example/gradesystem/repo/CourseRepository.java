package com.example.gradesystem.repo;

import com.example.gradesystem.model.Course;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class CourseRepository extends BaseTsvRepository<Course> {
    public CourseRepository(Path filePath) { super(filePath); }

    public List<Course> findAll() { return loadAll(); }

    public Optional<Course> findById(String id) {
        return loadAll().stream().filter(c -> c.getId().equals(id)).findFirst();
    }

    public void save(Course course) {
        List<Course> list = new ArrayList<>(loadAll());
        list.remove(course);
        list.add(course);
        saveAll(list);
    }

    public boolean deleteById(String id) {
        List<Course> list = new ArrayList<>(loadAll());
        boolean removed = list.removeIf(c -> c.getId().equals(id));
        if (removed) saveAll(list);
        return removed;
    }

    @Override protected Course parse(String line) {
        String[] p = line.split("\t", -1);
        return new Course(p[0], p[1], Integer.parseInt(p[2]));
    }

    @Override protected String serialize(Course item) {
        return item.getId() + "\t" + item.getName() + "\t" + item.getCredits();
    }
}
