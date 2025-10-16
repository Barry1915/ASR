package com.example.gradesystem.repo;

import com.example.gradesystem.model.Enrollment;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class EnrollmentRepository extends BaseTsvRepository<Enrollment> {
    public EnrollmentRepository(Path filePath) { super(filePath); }

    public List<Enrollment> findAll() { return loadAll(); }

    public List<Enrollment> findByStudentId(String studentId) {
        return loadAll().stream().filter(e -> e.getStudentId().equals(studentId)).collect(Collectors.toList());
    }

    public List<Enrollment> findByCourseId(String courseId) {
        return loadAll().stream().filter(e -> e.getCourseId().equals(courseId)).collect(Collectors.toList());
    }

    public Optional<Enrollment> find(String studentId, String courseId) {
        return loadAll().stream().filter(e -> e.getStudentId().equals(studentId) && e.getCourseId().equals(courseId)).findFirst();
    }

    public void save(Enrollment enrollment) {
        List<Enrollment> list = new ArrayList<>(loadAll());
        list.remove(enrollment);
        list.add(enrollment);
        saveAll(list);
    }

    @Override protected Enrollment parse(String line) {
        String[] p = line.split("\t", -1);
        Double grade = p[2].isBlank() ? null : Double.parseDouble(p[2]);
        return new Enrollment(p[0], p[1], grade);
    }

    @Override protected String serialize(Enrollment item) {
        String g = item.getGrade() == null ? "" : Double.toString(item.getGrade());
        return item.getStudentId() + "\t" + item.getCourseId() + "\t" + g;
    }
}
